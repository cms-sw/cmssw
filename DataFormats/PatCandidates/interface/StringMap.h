#ifndef DataFormats_PatCandidates_StringMap_h
#define DataFormats_PatCandidates_StringMap_h

#include <string>
#include <vector>
#include <algorithm>   // for std::pair

class StringMap {
    public:
        typedef std::pair<std::string, int32_t> value_type;
        typedef std::vector<value_type>         vector_type;
        typedef vector_type::const_iterator     const_iterator;

        void add(const std::string &string, int32_t value) ;
        void sort();
        void clear();

        /// return associated number, or -1 if no one is found
        /// in case the association is not unque, the choice of the returned value is undetermined
        /// note: works only after it's sorted
        int32_t operator[](const std::string &string) const ;

        /// return associated string, or "" if none is there
        /// in case the association is not unque, the choice of the returned value is undetermined
        /// note: works only after it's sorted
        const std::string & operator[](int32_t number) const ;

        const_iterator find(const std::string &string) const ;
        const_iterator find(int32_t number) const ;

        const_iterator begin() const { return entries_.begin(); }
        const_iterator end() const { return entries_.end(); }
    
        size_t size() const { return entries_.size(); }
        class MatchByString {
            public:
                MatchByString() {}
                //MatchByString(const std::string &string) : string_(string) {}
                bool operator()(const value_type &val, const std::string &string) const { return  val.first < string; }
                //bool operator()(const value_type &val) const { return string_ == val.first; }
            private:
                //const std::string &string_;
        };
        class MatchByNumber {
            public:
                MatchByNumber(int32_t number) : number_(number) {}
                bool operator()(const value_type &val) const { return number_ == val.second; }
            private:
                int32_t number_;
        };
    private:
        std::vector< std::pair<std::string, int32_t> > entries_;
        
} ;

#endif
