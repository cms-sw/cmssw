// original code by the Minima Authors under the MIT license. See AUTHORS for the list of authors.

#include "popenCPP.h"
#include "FWCore/Utilities/interface/Exception.h"
#include <cstdio>
#include <string>
#include <istream>


namespace{
        class fbuf final : public std::streambuf{
                FILE *file;
                char *buf;
                constexpr static size_t bufsz = 4096;
        public:
                fbuf(FILE *f);
                ~fbuf();
        protected:
                virtual std::streambuf *setbuf(char_type *s, std::streamsize n);
                virtual int sync();
                virtual int_type underflow();
        };

        class cfstream final : public std::istream{
                fbuf buf;
        public:
                cfstream(FILE *f);
        };
}

std::unique_ptr<std::istream> reco::exprEvalDetails::popenCPP(const std::string &cmdline){
        FILE *f = popen(cmdline.c_str(), "r");
        if(!f)
	  throw  cms::Exception("PopenCPP","(\""+cmdline+"\") failed");
        return std::unique_ptr<std::istream>(new cfstream(f));
}


fbuf::fbuf(FILE *f)
        : file(f), buf(new char[bufsz]){
        this->setg(this->buf, this->buf+bufsz, this->buf+bufsz);
}

fbuf::~fbuf(){
        delete [] this->buf;
        pclose(this->file);
}

std::streambuf *fbuf::setbuf(char_type *s, std::streamsize n){
        return nullptr;
}

int fbuf::sync(){
        if(fflush(this->file) != 0)
                return -1;
        return 0;
}

fbuf::int_type fbuf::underflow(){
        if(this->gptr() < this->egptr()){
                char c = *this->gptr();
                this->setg(this->eback(), this->gptr()+1, this->egptr());
                return traits_type::to_int_type(c);
        }
        size_t n = fread(this->buf, 1, sizeof(this->buf), this->file);
        if(n == 0)
                return traits_type::eof();
        this->setg(this->buf, this->buf, this->buf+n);
        return traits_type::to_int_type(*this->gptr());
}

cfstream::cfstream(FILE *f)
        : std::istream(&buf), buf(f){
}
