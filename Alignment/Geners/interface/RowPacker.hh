// A persistent tuple. Allows for type-safe access to its arguments.
// Optimized for a single row access (one tuple at a time).

#ifndef GENERS_ROWPACKER_HH_
#define GENERS_ROWPACKER_HH_

#include "Alignment/Geners/interface/CPP11_config.hh"
#ifdef CPP11_STD_AVAILABLE

#include <utility>

#include "Alignment/Geners/interface/AbsArchive.hh"
#include "Alignment/Geners/interface/ClassId.hh"
#include "Alignment/Geners/interface/CharBuffer.hh"
#include "Alignment/Geners/interface/RPHeaderRecord.hh"
#include "Alignment/Geners/interface/RPBufferRecord.hh"
#include "Alignment/Geners/interface/RPFooterRecord.hh"
#include "Alignment/Geners/interface/RPBufferReference.hh"
#include "Alignment/Geners/interface/RPReference.hh"

namespace gs {
    template<typename Pack>
    class RowPacker
    {
        template<typename Pack2> friend class RowPacker;
        static const unsigned defaultBufferSize = 800000U;

    public:
        typedef Pack value_type;

        // Comment on choosing the buffer size: this size should
        // be appropriate for the compression library used with
        // the archive, as each buffer is compressed independently.
        // For example, the bzip2 compression is done in chunks
        // which are slightly less than 900 kB (because in that
        // library 1 kB == 1000 bytes and because a small part of
        // the buffer is used for housekeeping). At the same time,
        // guaranteeing exact limit on the buffer size would slow
        // the code down: as the serialized size of each object
        // is not known in advance, each row would have to be
        // serialized into a separate buffer first instead of using
        // the common buffer. The code below utilizes a different
        // approach: the buffer can grow slightly above the limit,
        // and the buffer is dumped into the archive as soon as
        // its size exceeds the limit. Thus the actual size of the
        // archived buffer can be the value of "bufferSize"
        // parameter plus size of one serialized table row.
        // In principle, row size can be arbitrary, and this is why
        // the choice is left to the user.
        //
        // Naturally, setting the buffer size to 0 will result in
        // every row serialized and dumped to the archive as
        // a separate entity. This setting can have its own good
        // use if the expected row access pattern is random.
        //
        RowPacker(const std::vector<std::string>& columnNames,
                  const char* title, AbsArchive& archive,
                  const char* name, const char* category,
                  unsigned bufferSize = defaultBufferSize);

        // The following constructor will work only if every
        // element of the "protoPack" has a function with the
        // signature "const std::string& name() const" (think
        // gs::IOProxy).
        //
        RowPacker(const char* title, AbsArchive& archive,
                  const char* name, const char* category,
                  const Pack& protoPack,
                  unsigned bufferSize = defaultBufferSize);

        // A minimalistic constructor which can be used if you
        // do not care about things like column names and table
        // title. Default values will be assigned instead: all
        // columns will be named "c0", "c1", ..., and the title
        // will be an empty string.
        RowPacker(AbsArchive& archive,
                  const char* name, const char* category,
                  unsigned bufferSize = defaultBufferSize);

        inline ~RowPacker() {write();}

        // Various simple inspectors
        inline AbsArchive& archive() const {return ar_;}
        inline const std::string& name() const {return name_;}
        inline const std::string& category() const {return category_;}
        inline unsigned bufferSize() const {return bufferSize_;}
        inline bool isReadable() const {return readable_;}
        inline bool isWritable() const {return writable_;}
        inline const std::string& title() const {return title_;}

        // Each object of this type created in one particular program
        // will have its own unique number. This number in not persistent.
        inline unsigned long objectNumber() const {return objectNumber_;}

        // Simple modifiers
        inline void setTitle(const char* newtitle)
            {title_ = newtitle ? newtitle : "";}
        inline void setBufferSize(const unsigned newsize)
            {bufferSize_ = newsize;}

        // Dealing with rows and columns
        inline unsigned long nRows() const {return fillCount_;}

        inline unsigned long nColumns() const 
            {return std::tuple_size<Pack>::value;}

        inline const std::string& columnName(const unsigned long i) const
            {return colNames_.at(i);}

        inline const std::vector<std::string>& columnNames() const
            {return colNames_;}

        unsigned long columnNumber(const char* columnName) const;

        // The code will refuse to set the column name (and false will be
        // returned in this case) if the provided name duplicates an existing
        // column name. False will also be returned if the column index
        // is out of range.
        bool setColumnName(unsigned long i, const char* newname);

        // Fill one tuple. This method will return "true" on success and
        // "false" on failure.
        bool fill(const Pack& tuple);

        // Read the row contents back. This method will throw
        // gs::IOOutOfRange in case the row number is out of range.
        void rowContents(unsigned long row, Pack* tuple) const;

        template<typename Pack2>
        bool operator==(const RowPacker<Pack2>& r) const;

        template<typename Pack2>
        inline bool operator!=(const RowPacker<Pack2>& r) const
            {return !(*this == r);}

        // Methods needed for I/O
        bool write();
        inline ClassId classId() const {return ClassId(*this);}

        static const char* classname();
        static inline unsigned version() {return 1;}

    private:
        friend class Private::RPHeaderRecord<RowPacker<Pack> >;
        friend class Private::RPBufferRecord<RowPacker<Pack> >;
        friend class Private::RPFooterRecord<RowPacker<Pack> >;
        friend class Private::RPBufferReference<RowPacker<Pack> >;
        friend class RPReference<RowPacker<Pack> >;

        RowPacker();
        RowPacker(const RowPacker&);
        RowPacker& operator=(const RowPacker&);

        // The following function is used by RPReference
        static RowPacker* read(AbsArchive& ar,
                               std::istream& is,
                               unsigned long long headId);

        static unsigned long nextObjectNumber();

        void prepareToUnpack() const;
        bool unpackTuple(std::istream& is, Pack* tuple) const;
        std::istream& getRowStream(unsigned long row,
                                   unsigned long* len = 0) const;

        bool loadRowData(unsigned long rowNumber) const;
        void saveHeader();
        void saveFillBuffer();

        mutable CharBuffer fillBuffer_;
        unsigned long firstFillBufferRow_;
        std::vector<std::streampos> fillBufferOffsets_;

        mutable CharBuffer readBuffer_;
        mutable unsigned long firstReadBufferRow_;
        mutable std::vector<std::streampos> readBufferOffsets_;
        ClassId bufferClass_;
        ClassId thisClass_;

        AbsArchive& ar_;
        std::vector<std::string> colNames_;

        // The first member of the pair is the first row in the
        // buffer, the second is the id of the buffer in the archive
        std::vector<std::pair<unsigned long,unsigned long long> > idlist_;

        std::string name_;
        std::string category_;
        std::string title_;
        unsigned long long headerSaved_;
        unsigned long bufferSize_;
        unsigned long fillCount_;
        unsigned long objectNumber_;
        bool readable_;
        bool writable_;
        mutable bool firstUnpack_;
        bool unused_;
        mutable std::vector<std::vector<ClassId> > iostack_;

        static const std::vector<std::string>& defaultColumnNames();
    };
}

#include <memory>
#include <cstring>
#include <algorithm>

#include "Alignment/Geners/interface/tupleIO.hh"
#include "Alignment/Geners/interface/PackerIOCycle.hh"
#include "Alignment/Geners/interface/RPFooterReference.hh"
#include "Alignment/Geners/interface/collectTupleNames.hh"
#include "Alignment/Geners/interface/allUnique.hh"
#include "Alignment/Geners/interface/findName.hh"
#include "Alignment/Geners/interface/IOException.hh"

namespace gs {
    template <typename T>
    unsigned long RowPacker<T>::nextObjectNumber()
    {
        static unsigned long ocounter = 0;
        return ocounter++;
    }


    template <typename T>
    unsigned long RowPacker<T>::columnNumber(const char* columnName) const
    {
        return findName(colNames_, columnName);
    }


    template<typename T>
    inline RowPacker<T>::RowPacker(const std::vector<std::string>& colNames,
                                   const char* ititle, AbsArchive& iarchive,
                                   const char* iname, const char* icategory,
                                   const unsigned ibufferSize)
        : firstFillBufferRow_(0),
          firstReadBufferRow_(0),
          bufferClass_(fillBuffer_.classId()),
          thisClass_(ClassId::makeId<RowPacker<T> >()),
          ar_(iarchive),
          colNames_(colNames),
          name_(iname ? iname : ""),
          category_(icategory ? icategory : ""),
          title_(ititle ? ititle : ""),
          headerSaved_(0),
          bufferSize_(ibufferSize),
          fillCount_(0),
          objectNumber_(nextObjectNumber()),
          readable_(ar_.isReadable()),
          writable_(ar_.isWritable()),
          firstUnpack_(true)
    {
        if (!std::tuple_size<T>::value) throw gs::IOInvalidArgument(
            "In RowPacker constructor: can not use empty tuple");
        if (std::tuple_size<T>::value != colNames.size())
            throw gs::IOInvalidArgument("In RowPacker constructor: "
                                        "wrong # of column names");
        if (!allUnique(colNames_)) throw gs::IOInvalidArgument(
            "In RowPacker constructor: all column names must be unique");
    }


    template<typename T>
    inline RowPacker<T>::RowPacker(const char* ititle, AbsArchive& iarchive,
                                   const char* iname, const char* icategory,
                                   const T& protoPack,
                                   const unsigned ibufferSize)
        : firstFillBufferRow_(0),
          firstReadBufferRow_(0),
          bufferClass_(fillBuffer_.classId()),
          thisClass_(ClassId::makeId<RowPacker<T> >()),
          ar_(iarchive),
          colNames_(collectTupleNames(protoPack)),
          name_(iname ? iname : ""),
          category_(icategory ? icategory : ""),
          title_(ititle ? ititle : ""),
          headerSaved_(0),
          bufferSize_(ibufferSize),
          fillCount_(0),
          objectNumber_(nextObjectNumber()),
          readable_(ar_.isReadable()),
          writable_(ar_.isWritable()),
          firstUnpack_(true)
    {
        if (!std::tuple_size<T>::value) throw gs::IOInvalidArgument(
            "In RowPacker constructor: can not use empty tuple");
        if (!allUnique(colNames_)) throw gs::IOInvalidArgument(
            "In RowPacker constructor: all column names must be unique");
    }


    // The compiler that I have now still does not support
    // constructor delegation...
    template <typename T>
    inline RowPacker<T>::RowPacker(AbsArchive& iarchive,
                                   const char* iname, const char* icategory,
                                   const unsigned ibufferSize)
        : firstFillBufferRow_(0),
          firstReadBufferRow_(0),
          bufferClass_(fillBuffer_.classId()),
          thisClass_(ClassId::makeId<RowPacker<T> >()),
          ar_(iarchive),
          colNames_(defaultColumnNames()),
          name_(iname ? iname : ""),
          category_(icategory ? icategory : ""),
          title_(""),
          headerSaved_(0),
          bufferSize_(ibufferSize),
          fillCount_(0),
          objectNumber_(nextObjectNumber()),
          readable_(ar_.isReadable()),
          writable_(ar_.isWritable()),
          firstUnpack_(true)
    {
        if (!std::tuple_size<T>::value) throw gs::IOInvalidArgument(
            "In RowPacker constructor: can not use empty tuple");
    }


    template <typename T>
    inline void RowPacker<T>::saveHeader()
    {
        if (!headerSaved_ && writable_)
        {
            Private::RPHeaderRecord<RowPacker<T> > record(*this);
            ar_ << record;
            headerSaved_ = record.id();

            // record id must be positive (or operator << used above must
            // throw an exception)
            assert(headerSaved_);
        }
    }


    template <typename T>
    void RowPacker<T>::saveFillBuffer()
    {
        saveHeader();
        Private::RPBufferRecord<RowPacker<T> > record(*this);
        ar_ << record;
        idlist_.push_back(std::make_pair(firstFillBufferRow_, record.id()));
        fillBuffer_.seekp(0);
        firstFillBufferRow_ = fillCount_;
        fillBufferOffsets_.clear();
    }


    template <typename Pack>
    bool RowPacker<Pack>::fill(const Pack& tuple)
    {
        if (!writable_)
            return false;

        if (fillBuffer_.size() > bufferSize_)
            saveFillBuffer();

        fillBufferOffsets_.push_back(fillBuffer_.tellp());
        const bool wstatus = write_item(fillBuffer_, tuple, false);
        if (wstatus)
            ++fillCount_;
        return wstatus;
    }


    template <typename T>
    const char* RowPacker<T>::classname()
    {
        static const std::string myClass(
            template_class_name<T>("gs::RowPacker"));
        return myClass.c_str();
    }


    template <typename T>
    bool RowPacker<T>::write()
    {
        if (!writable_)
            return false;
        saveHeader();
        if (!fillBufferOffsets_.empty())
            saveFillBuffer();
        ar_ << Private::RPFooterRecord<RowPacker<T> >(*this);
        writable_ = false;
        return true;
    }

    // The following function returns "true" if the row
    // is inside fillBuffer_, and "false" if it is inside
    // "readBuffer_".
    template <typename T>
    bool RowPacker<T>::loadRowData(const unsigned long rowNumber) const
    {
        assert(readable_);
        if (rowNumber >= fillCount_)
            throw gs::IOOutOfRange("In RowPacker::loadRowData: "
                                    "row number is out of range");
        if (rowNumber >= firstFillBufferRow_ &&
            rowNumber < firstFillBufferRow_ + fillBufferOffsets_.size())
            return true;
        if (rowNumber >= firstReadBufferRow_ &&
            rowNumber < firstReadBufferRow_ + readBufferOffsets_.size())
            return false;

        // The row number is not in any buffer. We need to load
        // the row data from the archive.
        const unsigned long nSaved = idlist_.size();
        unsigned long bucket = std::lower_bound(
            idlist_.begin(), idlist_.end(), std::make_pair(rowNumber, 0ULL)) -
            idlist_.begin();
        if (bucket == nSaved)
            --bucket;
        else if (idlist_[bucket].first != rowNumber)
            --bucket;
        const Private::RPBufferReference<RowPacker<T> >& ref = 
            Private::RPBufferReference<RowPacker<T> >(
                *this, idlist_.at(bucket).second);
        ref.restore(0);
        return false;
    }


    template <typename Pack>
    void RowPacker<Pack>::prepareToUnpack() const
    {
        // Prepare proper I/O stack
        std::vector<std::vector<ClassId> > dummy;
        thisClass_.templateParameters(&dummy);
        assert(dummy.size() == 1U);
        dummy[0].at(0).templateParameters(&iostack_);
        assert(iostack_.size() == std::tuple_size<Pack>::value);
    }


    template <typename Pack>
    inline bool RowPacker<Pack>::unpackTuple(
        std::istream& is, Pack* pack) const
    {
        if (firstUnpack_)
        {
            prepareToUnpack();
            firstUnpack_ = false;
        }
        return Private::PackerIOCycle<Pack,std::tuple_size<Pack>::value>::read(
            pack, is, iostack_);
    }


    template <typename T>
    std::istream& RowPacker<T>::getRowStream(
        const unsigned long row, unsigned long* len) const
    {
        if (loadRowData(row))
        {
            // Fill buffer
            const unsigned long idx = row - firstFillBufferRow_;
            fillBuffer_.clear();
            fillBuffer_.seekg(fillBufferOffsets_.at(idx));
            assert(!fillBuffer_.fail());
            if (len)
            {
                const unsigned long thispos = fillBufferOffsets_[idx];
                if (idx + 1UL < fillBufferOffsets_.size())
                {
                    const unsigned long nextpos = fillBufferOffsets_[idx+1UL];
                    *len = nextpos - thispos;
                }
                else
                    *len = fillBuffer_.size() - thispos;
            }
            return fillBuffer_;
        }
        else
        {
            // Read buffer
            const unsigned long idx = row - firstReadBufferRow_;
            readBuffer_.clear();
            readBuffer_.seekg(readBufferOffsets_.at(idx));
            assert(!readBuffer_.fail());
            if (len)
            {
                const unsigned long thispos = readBufferOffsets_[idx];
                if (idx + 1UL < readBufferOffsets_.size())
                {
                    const unsigned long nextpos = readBufferOffsets_[idx+1UL];
                    *len = nextpos - thispos;
                }
                else
                    *len = readBuffer_.size() - thispos;
            }
            return readBuffer_;
        }
    }


    template <typename T>
    void RowPacker<T>::rowContents(const unsigned long row, T* tuple) const
    {
        if (row < fillCount_)
        {
            assert(tuple);
            if (!unpackTuple(getRowStream(row), tuple))
                throw IOInvalidData("In gs::RowPacker::rowContents: "
                                    "failed to unpack tuple data");
        }
        else
            throw gs::IOOutOfRange("In gs::RowPacker::rowContents: "
                                    "row number is out of range");
    }


    template <typename T>
    template <typename Pack2>
    bool RowPacker<T>::operator==(const RowPacker<Pack2>& r) const
    {
        if ((void *)this == (void *)(&r))
            return true;
        if (!readable_ || !r.readable_)
            return false;
        if (nColumns() != r.nColumns())
            return false;
        if (fillCount_ != r.fillCount_)
            return false;
        if (thisClass_.name() != r.thisClass_.name())
            return false;
        if (title_ != r.title_)
            return false;
        if (colNames_ != r.colNames_)
            return false;
        for (unsigned long row=0; row<fillCount_; ++row)
        {
            unsigned long len1=0, len2=0;
            std::istream& s1 = getRowStream(row, &len1);
            std::istream& s2 = r.getRowStream(row, &len2);
            if (len1 != len2)
                return false;
            std::streambuf* buf1 = s1.rdbuf();
            std::streambuf* buf2 = s2.rdbuf();
            unsigned long i=0;
            for (; i<len1 && buf1->sbumpc() == buf2->sbumpc(); ++i) {;}
            if (i < len1) return false;
        }
        return true;
    }


    template <typename T>
    RowPacker<T>* RowPacker<T>::read(AbsArchive& ar,
                                     std::istream& is,
                                     const unsigned long long headerId)
    {
        static const ClassId current(ClassId::makeId<RowPacker<T> >());

        if (!ar.isReadable()) throw gs::IOInvalidArgument(
            "In RowPacker::read: archive not readable");
        if (!headerId) throw gs::IOInvalidArgument(
            "In RowPacker::read: invalid header record id");
        std::shared_ptr<const CatalogEntry> headerRecord = 
            ar.catalogEntry(headerId);
        if (!headerRecord.get())  throw gs::IOInvalidArgument(
            "In RowPacker::read: header record not found");
        if (headerRecord->id() != headerId) throw IOInvalidData(
            "In RowPacker::read: recorded header id does not match catalog id");

        // Unpack in the pack order of RPHeaderRecord
        ClassId packerClass(is, 1);
        current.ensureSameName(packerClass);

        ClassId bufferClass(is, 1);
        std::vector<std::string> columnNames;
        read_pod_vector(is, &columnNames);
        std::string titl;
        read_pod(is, &titl);
        unsigned long bufSiz;
        read_pod(is, &bufSiz);
        if (is.fail())
            throw IOReadFailure("In RowPacker::read: input stream failure");

        // Now we have all the info from the header record.
        // Find the corresponding footer.
        Private::RPFooterReference footerInfo(
            ar, packerClass, headerRecord->name().c_str(),
            headerRecord->category().c_str());
        const unsigned long nfooters = footerInfo.size();

        // We should be able to handle situations with
        // a missing footer. Code this when time permits.
        if (nfooters == 0)
            throw IOInvalidData("In RowPacker::read: footer record not found");

        unsigned long nrows = 0;
        unsigned long long writtenHeaderId = 0;
        std::vector<std::pair<unsigned long,unsigned long long> > bufferIds;
        unsigned long long offset = 0;

        // Find the right footer if we have more than one of them
        for (unsigned long ifoot = 0; ifoot < nfooters; ++ifoot)
        {
            footerInfo.fillItems(
                &nrows, &writtenHeaderId, &bufferIds, &offset, ifoot);
            if (writtenHeaderId + offset == headerId)
                break;
        }
        if (writtenHeaderId + offset != headerId)
            throw IOInvalidData("In RowPacker::read: "
                                "incompatible footer record");

        // Recalculate the buffer ids
        if (offset)
        {
            const unsigned long nbuf = bufferIds.size();
            for (unsigned long ibuf=0; ibuf<nbuf; ++ibuf)
                bufferIds[ibuf].second += offset;
        }

        // Finally, build the packer
        std::unique_ptr<RowPacker<T> > nt(new RowPacker<T>(
            columnNames, titl.c_str(), ar, headerRecord->name().c_str(),
            headerRecord->category().c_str(), bufSiz));

        nt->bufferClass_ = bufferClass;
        nt->thisClass_ = packerClass;
        nt->headerSaved_ = writtenHeaderId;
        nt->fillCount_ = nrows;
        nt->writable_ = false;

        const bool buffersEmpty = bufferIds.empty();
        if (!(nrows ? !buffersEmpty : buffersEmpty))
            throw IOInvalidData("In RowPacker::read: "
                                "corrupted footer record");
        if (nrows)
        {
            nt->idlist_ = bufferIds;
            nt->firstFillBufferRow_ = nrows;
        }

        return nt.release();
    }


    template<typename Pack>
    inline const std::vector<std::string>&
    RowPacker<Pack>::defaultColumnNames()
    {
        return default_tuple_columns<std::tuple_size<Pack>::value>();
    }


    template<typename Pack>
    bool RowPacker<Pack>::setColumnName(unsigned long i, const char* newname)
    {
        const unsigned long n = colNames_.size();
        if (i >= n)
            return false;
        if (columnNumber(newname) < n)
            return false;
        colNames_[i] = newname;
        return true;
    }
}


#endif // CPP11_STD_AVAILABLE
#endif // GENERS_ROWPACKER_HH_

