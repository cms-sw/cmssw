// A persistent tuple. Allows for type-safe access to its arguments.
// Optimized for column access.

#ifndef GENERS_COLUMNPACKER_HH_
#define GENERS_COLUMNPACKER_HH_

#include "Alignment/Geners/interface/CPP11_config.hh"
#ifdef CPP11_STD_AVAILABLE

#include "Alignment/Geners/interface/ColumnPackerHelper.hh"
#include "Alignment/Geners/interface/CPHeaderRecord.hh"
#include "Alignment/Geners/interface/CPFooterRecord.hh"
#include "Alignment/Geners/interface/CPReference.hh"
#include "Alignment/Geners/interface/tupleIO.hh"

namespace gs {
    template<typename Pack>
    class ColumnPacker
    {
        template<typename Pack2> friend class ColumnPacker;

        // Each column will have its own write (and possibly read) buffer,
        // so the default buffer size should not be too large
        static const unsigned defaultBufferSize = 65536U;

    public:
        typedef Pack value_type;

        ColumnPacker(const std::vector<std::string>& columnNames,
                     const char* title, AbsArchive& archive,
                     const char* name, const char* category,
                     unsigned bufferSize = defaultBufferSize);

        // The following constructor will work only if every
        // element of the "protoPack" has a function with the
        // signature "const std::string& name() const" (think
        // gs::IOProxy).
        //
        ColumnPacker(const char* title, AbsArchive& archive,
                     const char* name, const char* category,
                     const Pack& protoPack,
                     unsigned bufferSize = defaultBufferSize);

        // A minimalistic constructor which can be used if you
        // do not care about things like column names and table
        // title. Default values will be assigned instead: all
        // columns will be named "c0", "c1", ..., and the title
        // will be an empty string.
        ColumnPacker(AbsArchive& archive,
                     const char* name, const char* category,
                     unsigned bufferSize = defaultBufferSize);

        ~ColumnPacker();

        // Various simple inspectors
        inline AbsArchive& archive() const {return ar_;}
        inline const std::string& name() const {return name_;}
        inline const std::string& category() const {return category_;}
        inline unsigned long bufferSize() const {return bufferSize_;}
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

        // The following method will return "true" if all column names
        // and types of this tuple correspond to column names and types
        // of the archived tuple (and come in the same order).
        inline bool isOriginal() const {return isOriginalTuple_;}

        // Info about columns of the original tuple saved in the archive
        unsigned long nOriginalColumns() const;
        const std::string& originalColumnName(unsigned long i) const;
        const std::vector<std::string>& originalColumnNames() const;
        unsigned long originalColumnNumber(const char* columnName) const;

        // A faster way to calculate the original column number
        // which corresponds to the current column number. Returns
        // nOriginalColumns() for columns which were not mapped or
        // if the argument column number is out of range.
        unsigned long originalColumn(unsigned long currentNumber) const;

        // Number of columns which will be read when "rowContents"
        // function is called
        unsigned long nColumnsReadBack() const;

        // The following function returns the mask of tuple elements
        // which will be filled on readback (1 for filled, 0 for not
        // filled).
        inline const std::vector<unsigned char>& readbackMask() const
            {return readbackMask_;}

        // Fill one tuple. This method returns "true" is everything is fine,
        // "false" if there was a problem.
        bool fill(const Pack& tuple);

        // Read the row contents back. This method will throw
        // gs::IOOutOfRange in case the row number is out of range.
        void rowContents(unsigned long row, Pack* tuple) const;

        // Fetch just one item at the given row and column. "StoragePtr"
        // should be an appropriate pointer type: IOPtr, IOProxy, bare
        // pointer, or CPP11_shared_ptr, depending on how the data was
        // actually packed. The signature of this method is kind of ugly,
        // so it may be more convenient to use column iterators.
        //
        // This method will throw gs::IOOutOfRange in case the row number
        // is out of range and gs::IOInvalidArgument if the column was
        // disabled when the packer was read back from the archive.
        //
        template <unsigned long Column, typename StoragePtr>
        void fetchItem(unsigned long row, StoragePtr ptr) const;

        // Compare packed contents
        template<typename Pack2>
        bool operator==(const ColumnPacker<Pack2>& r) const;

        template<typename Pack2>
        inline bool operator!=(const ColumnPacker<Pack2>& r) const
            {return !(*this == r);}

        // Methods needed for I/O
        bool write();
        inline ClassId classId() const {return ClassId(*this);}

        static const char* classname();
        static inline unsigned version() {return 1;}

    private:
        template<typename Pack2, int N>
        friend class Private::ColumnPackerHelper;
        friend class Private::CPHeaderRecord<ColumnPacker<Pack> >;
        friend class Private::CPFooterRecord<ColumnPacker<Pack> >;
        friend class CPReference<ColumnPacker<Pack> >;

        ColumnPacker();
        ColumnPacker(const ColumnPacker&);
        ColumnPacker& operator=(const ColumnPacker&);

        // The following function is used by CPReference
        static ColumnPacker* read(
            AbsArchive& ar, std::istream& is,
            unsigned long long headId,
            const std::vector<std::string>& colNames,
            bool namesProvided);

        static unsigned long nextObjectNumber();

        AbsArchive& ar_;
        std::vector<std::string> colNames_;

        std::string name_;
        std::string category_;
        std::string title_;
        unsigned long long headerSaved_;
        unsigned long bufferSize_;
        unsigned long fillCount_;
        const unsigned long objectNumber_;
        bool readable_;
        bool writable_;
        mutable bool firstUnpack_;

        // The following flag may be set to "false" by the "read"
        // function
        bool isOriginalTuple_;

        std::vector<Private::ColumnBuffer*> fillBuffers_;
        mutable std::vector<Private::ColumnBuffer*> readBuffers_;
        mutable std::vector<std::vector<ClassId> > iostack_;
        mutable unsigned long currentReadRow_;
        std::vector<unsigned char> readbackMask_;

        // Id list for the buffers. The fisrt index is the column number.
        // The internal vector contains the pairs of the starting row
        // number and the buffer id in the archive.
        std::vector<std::vector<std::pair<
            unsigned long,unsigned long long> > > bufIds_;

        ClassId thisClass_;
        ClassId bufferClass_;
        ClassId cbClass_;

        // The vector of original column names. May or may not
        // be present, depending on how ntuple was created.
        std::vector<std::string>* originalColNames_;

        // The vector of original column numbers. May or may not
        // be present, depending on how ntuple was created.
        std::vector<unsigned long>* originalColNumber_;

        void initialize();
        void saveHeader();
        void saveFillBuffer(unsigned long col);
        void prepareToUnpack() const;

        inline bool dumpColumnClassIds(std::ostream& os) const
        {
            return Private::TupleClassIdCycler<
                Pack,std::tuple_size<Pack>::value>::dumpClassIds(os);
        }

        std::istream* getRowStream(
            unsigned long row, unsigned long col, unsigned long* len=0) const;

        std::ostream& columnOstream(unsigned long columnNumber);

        std::istream* columnIstream(unsigned long columnNumber,
                                    std::vector<ClassId>** iostack) const;

        static const std::vector<std::string>& defaultColumnNames();
    };
}

#include <memory>
#include <algorithm>
#include <numeric>
#include <cassert>

#include "Alignment/Geners/interface/allUnique.hh"
#include "Alignment/Geners/interface/CPBufferRecord.hh"
#include "Alignment/Geners/interface/CPBufferReference.hh"
#include "Alignment/Geners/interface/CPFooterReference.hh"
#include "Alignment/Geners/interface/findName.hh"
#include "Alignment/Geners/interface/IOIsSameType.hh"

#include "Alignment/Geners/interface/IOException.hh"

namespace gs {
    template <typename T>
    unsigned long ColumnPacker<T>::nextObjectNumber()
    {
        static unsigned long ocounter = 0;
        return ocounter++;
    }


    template <typename T>
    void ColumnPacker<T>::initialize()
    {
        const unsigned long nCols = std::tuple_size<T>::value;
        if (!nCols) throw gs::IOInvalidArgument(
            "In ColumnPacker::initialize: can not use empty tuple");
        if (nCols != colNames_.size()) throw gs::IOInvalidArgument(
            "In ColumnPacker::initialize: wrong # of column names");
        if (!allUnique(colNames_)) throw gs::IOInvalidArgument(
            "In ColumnPacker::initialize: all column names must be unique");

        // Initialize buffers.
        fillBuffers_.resize(nCols);
        for (unsigned long i=0; i<nCols; ++i)
            fillBuffers_[i] = new Private::ColumnBuffer();

        bufIds_.resize(nCols);

        // Figure out PODness of the columns.
        Private::ColumnPackerHelper<
            T,std::tuple_size<T>::value>::podness(fillBuffers_);
    }


    template <typename T>
    inline ColumnPacker<T>::ColumnPacker(
        const std::vector<std::string>& colNames,
        const char* ititle, AbsArchive& iarchive,
        const char* iname, const char* icategory,
        const unsigned ibufferSize)
        : ar_(iarchive),
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
          firstUnpack_(true),
          isOriginalTuple_(true),
          readbackMask_(std::tuple_size<T>::value, 1),
          thisClass_(ClassId::makeId<ColumnPacker<T> >()),
          bufferClass_(ClassId::makeId<Private::ColumnBuffer>()),
          cbClass_(ClassId::makeId<CharBuffer>()),
          originalColNames_(0),
          originalColNumber_(0)
    {
        initialize();
    }


    template <typename T>
    inline ColumnPacker<T>::ColumnPacker(
        const char* ititle, AbsArchive& iarchive,
        const char* iname, const char* icategory,
        const T& protoPack,
        const unsigned ibufferSize)
        : ar_(iarchive),
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
          firstUnpack_(true),
          isOriginalTuple_(true),
          readbackMask_(std::tuple_size<T>::value, 1),
          thisClass_(ClassId::makeId<ColumnPacker<T> >()),
          bufferClass_(ClassId::makeId<Private::ColumnBuffer>()),
          cbClass_(ClassId::makeId<CharBuffer>()),
          originalColNames_(0),
          originalColNumber_(0)
    {
        initialize();
    }


    template <typename T>
    inline ColumnPacker<T>::ColumnPacker(
        AbsArchive& iarchive,
        const char* iname, const char* icategory,
        const unsigned ibufferSize)
        : ar_(iarchive),
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
          firstUnpack_(true),
          isOriginalTuple_(true),
          readbackMask_(std::tuple_size<T>::value, 1),
          thisClass_(ClassId::makeId<ColumnPacker<T> >()),
          bufferClass_(ClassId::makeId<Private::ColumnBuffer>()),
          cbClass_(ClassId::makeId<CharBuffer>()),
          originalColNames_(0),
          originalColNumber_(0)
    {
        initialize();
    }


    template <typename T>
    ColumnPacker<T>* ColumnPacker<T>::read(
        AbsArchive& ar, std::istream& is,
        const unsigned long long headerId,
        const std::vector<std::string>& colNamesIn,
        const bool colNamesProvided)
    {
        if (!ar.isReadable()) throw gs::IOInvalidArgument(
            "In ColumnPacker::read: archive not readable");
        if (!headerId) throw gs::IOInvalidArgument(
            "In ColumnPacker::read: invalid header record id");
        std::shared_ptr<const CatalogEntry> headerRecord = 
            ar.catalogEntry(headerId);
        if (!headerRecord.get())  throw gs::IOInvalidArgument(
            "In ColumnPacker::read: header record not found");
        if (headerRecord->id() != headerId) throw IOInvalidData(
            "In ColumnPacker::read: recorded header id "
            "does not match catalog id");

        const unsigned long ncols = std::tuple_size<T>::value;

        // Unpack in the pack order of CPHeaderRecord
        ClassId packerClass(is, 1);
        ClassId bufferClass(is, 1);
        ClassId cbClass(is, 1);

        std::vector<std::string> originalColumns;
        read_pod_vector(is, &originalColumns);
        std::string titl;
        read_pod(is, &titl);
        unsigned long bufSiz;
        read_pod(is, &bufSiz);
        if (is.fail())
            throw IOReadFailure("In ColumnPacker::read: input stream failure");

        // Read the column class ids
        std::vector<ClassId> origColClassIds;
        const unsigned long nColsHead = originalColumns.size();
        origColClassIds.reserve(nColsHead);
        for (unsigned long icol=0; icol<nColsHead; ++icol)
        {
            ClassId clid(is, 1);
            origColClassIds.push_back(clid);
        }
        if (nColsHead != origColClassIds.size())
            throw IOInvalidData("In ColumnPacker::read: "
                                "corrupted header record");

        // If the column names are not provided, make sure
        // we can pick up the right number of names from
        // the archive.
        if (colNamesProvided)
        {
            if (ncols != colNamesIn.size())
                throw gs::IOInvalidArgument("In ColumnPacker::read: "
                                            "wrong # of column names");
        }
        else
        {
            if (ncols != originalColumns.size())
                throw IOInvalidData("In ColumnPacker::read: incompatible # "
                                    "of columns on record");
        }

        // Find the corresponding footer
        Private::CPFooterReference footerInfo(
            ar, packerClass, headerRecord->name().c_str(),
            headerRecord->category().c_str());
        const unsigned long nfooters = footerInfo.size();

        // We should be able to handle situations with
        // a missing footer. Code this when time permits.
        if (nfooters == 0)
            throw IOInvalidData("In ColumnPacker::read: "
                                "footer record not found");

        unsigned long nrows = 0;
        unsigned long long writtenHeaderId = 0;
        std::vector<std::vector<std::pair<
            unsigned long,unsigned long long> > > bufferIds;
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
            throw IOInvalidData("In ColumnPacker::read: "
                                "incompatible footer record");
        if (nColsHead != bufferIds.size())
            throw IOInvalidData("In ColumnPacker::read: "
                                "corrupted footer record");

        // Recalculate the buffer ids
        if (offset)
            for (unsigned long icol=0; icol<nColsHead; ++icol)
            {
                std::vector<std::pair<
                    unsigned long,unsigned long long> >& bvec(bufferIds[icol]);
                const unsigned long nbuf = bvec.size();
                for (unsigned long ibuf=0; ibuf<nbuf; ++ibuf)
                    bvec[ibuf].second += offset;
            }

        // Now we can build the packer
        const std::vector<std::string>& colNames(
            colNamesProvided ? colNamesIn : originalColumns);
        ColumnPacker<T>* nt = new ColumnPacker<T>(
            colNames, titl.c_str(), ar, headerRecord->name().c_str(),
            headerRecord->category().c_str(), bufSiz);

        nt->thisClass_ = packerClass;
        nt->bufferClass_ = bufferClass;
        nt->cbClass_ = cbClass;
        nt->headerSaved_ = writtenHeaderId;
        nt->fillCount_ = nrows;
        nt->writable_ = false;        

        if (nrows)
            for (unsigned long icol=0; icol<ncols; ++icol)
            {
                Private::ColumnBuffer& buf(*nt->fillBuffers_[icol]);
                buf.firstrow = nrows;
                buf.lastrowp1 = nrows;
            }

        // Get the vector of class ids for the new tuple
        std::vector<ClassId> newClassIds;
        newClassIds.reserve(ncols);
        Private::TupleClassIdCycler<
            T,std::tuple_size<T>::value>::fillClassIdVector(&newClassIds);

        // Figure out the column mapping for readback
        std::vector<std::string>::iterator obeg = originalColumns.begin();
        std::vector<std::string>::iterator oend = originalColumns.end();
        std::vector<unsigned long> columnMap(ncols, originalColumns.size());
        bool allMatched = true;
        for (unsigned long icol=0; icol<ncols; ++icol)
        {
            bool matchFound = false;
            std::vector<std::string>::iterator it = 
                colNamesProvided ? std::find(obeg, oend, colNames[icol]) :
                obeg + icol;
            if (it != oend)
            {
                // There is a name match, but we still need to
                // make sure that the new column has a correct type
                const unsigned long idx = it - obeg;
                if (newClassIds[icol].name() == origColClassIds[idx].name())
                {
                    matchFound = true;

                    // Transfer the information about archive buffer ids
                    // to the new column
                    nt->bufIds_[icol] = bufferIds[idx];
                    const bool empty = nt->bufIds_[icol].empty();
                    if (!(nrows ? !empty : empty))
                        throw IOInvalidData("In ColumnPacker::read: "
                                            "corrupted footer record");
                    columnMap[icol] = idx;
                }
            }
            if (!matchFound)
            {
                // Disable the readback for this column
                nt->readbackMask_[icol] = 0;
                allMatched = false;
            }
        }

        if (!(allMatched && originalColumns == colNames))
        {
            nt->isOriginalTuple_ = false;
            nt->originalColNames_ = new std::vector<std::string>(
                originalColumns);
            nt->originalColNumber_ = new std::vector<unsigned long>(columnMap);
        }

        return nt;
    }


    template <typename T>
    ColumnPacker<T>::~ColumnPacker()
    {
        write();

        const unsigned long n1 = fillBuffers_.size();
        for (unsigned long i=0; i<n1; ++i)
            delete fillBuffers_[i];

        const unsigned long n2 = readBuffers_.size();
        for (unsigned long i=0; i<n2; ++i)
            delete readBuffers_[i];

        delete originalColNames_;
        delete originalColNumber_;
    }


    template <typename T>
    void ColumnPacker<T>::saveHeader()
    {
        if (!headerSaved_ && writable_)
        {
            Private::CPHeaderRecord<ColumnPacker<T> > record(*this);
            ar_ << record;
            headerSaved_ = record.id();
            assert(headerSaved_);
        }
    }


    template <typename T>
    void ColumnPacker<T>::saveFillBuffer(const unsigned long col)
    {
        saveHeader();
        Private::ColumnBuffer& buf(*fillBuffers_[col]);
        Private::CPBufferRecord record(buf, name_.c_str(),
                                       category_.c_str(), col);
        ar_ << record;
        bufIds_[col].push_back(std::make_pair(buf.firstrow, record.id()));
        buf.buf.seekp(0);
        buf.firstrow = fillCount_;
        buf.lastrowp1 = fillCount_;
        buf.offsets.clear();
    }


    template <typename Pack>
    std::ostream& ColumnPacker<Pack>::columnOstream(const unsigned long col)
    {
        Private::ColumnBuffer& buffer(*fillBuffers_[col]);
        if (buffer.buf.size() > bufferSize_)
            saveFillBuffer(col);
        if (!buffer.podsize)
            buffer.offsets.push_back(buffer.buf.tellp());
        ++buffer.lastrowp1;
        return buffer.buf;
    }


    template <typename Pack>
    std::istream* ColumnPacker<Pack>::getRowStream(
        const unsigned long row, const unsigned long col,
        unsigned long* len) const
    {
        Private::ColumnBuffer* goodbuf = 0;
        {
            Private::ColumnBuffer& fbuf(*fillBuffers_[col]);
            if (row >= fbuf.firstrow && row < fbuf.lastrowp1)
                goodbuf = &fbuf;
        }
        if (goodbuf == 0)
        {
            Private::ColumnBuffer& fbuf(*readBuffers_[col]);
            goodbuf = &fbuf;
            if (!(row >= fbuf.firstrow && row < fbuf.lastrowp1))
            {
                // Have to load the buffer from the archive
                assert(readable_);
                const std::vector<std::pair<
                    unsigned long,unsigned long long> >& idlist(bufIds_[col]);
                const unsigned long nSaved = idlist.size();
                unsigned long bucket = std::lower_bound(
                    idlist.begin(), idlist.end(), std::make_pair(row, 0ULL)) -
                    idlist.begin();
                if (bucket == nSaved)
                    --bucket;
                else if (idlist[bucket].first != row)
                    --bucket;
                const Private::CPBufferReference& ref = 
                    Private::CPBufferReference(ar_, bufferClass_, cbClass_,
                                               idlist.at(bucket).second);
                if (!ref.unique())
                    throw IOInvalidData("In gs::ColumnPacker::getRowStream: "
                                  "failed to obtain unique buffer reference");
                unsigned long readbackCol = 0;
                const unsigned long expectedCol = originalColumn(col);
                ref.restore(0, &fbuf, &readbackCol);
                if (!(readbackCol == expectedCol &&
                      row >= fbuf.firstrow && row < fbuf.lastrowp1))
                    throw IOInvalidData("In gs::ColumnPacker::getRowStream: "
                                  "failed to fetch the data from the archive");
            }
        }

        const unsigned long idx = row - goodbuf->firstrow;
        goodbuf->buf.clear();
        if (goodbuf->podsize)
            goodbuf->buf.seekg(idx*goodbuf->podsize);
        else
            goodbuf->buf.seekg(goodbuf->offsets.at(idx));
        assert(!goodbuf->buf.fail());

        if (len)
        {
            if (goodbuf->podsize)
                *len = goodbuf->podsize;
            else
            {
                const unsigned long thispos = goodbuf->offsets[idx];
                if (row + 1UL < goodbuf->lastrowp1)
                {
                    const unsigned long nextpos = goodbuf->offsets[idx+1UL];
                    *len = nextpos - thispos;
                }
                else
                    *len = goodbuf->buf.size() - thispos;
            }
        }

        return &goodbuf->buf;
    }


    template <typename Pack>
    std::istream* ColumnPacker<Pack>::columnIstream(
        const unsigned long col, std::vector<ClassId>** iostack) const
    {
        if (readbackMask_[col])
        {
            *iostack = &iostack_[col];
            return getRowStream(currentReadRow_, col);
        }
        return 0;
    }


    template <typename Pack>
    bool ColumnPacker<Pack>::fill(const Pack& tuple)
    {
        if (!writable_)
            return false;

        const bool status = Private::ColumnPackerHelper<
                   Pack,std::tuple_size<Pack>::value>::write(*this, tuple);
        if (!status)
            throw IOWriteFailure("In ColumnPacker::fill: tuple write failure. "
                                 "The record is likely to be corrupted now.");
        ++fillCount_;
        return status;
    }


    template <typename Pack>
    void ColumnPacker<Pack>::prepareToUnpack() const
    {
        // Prepare proper I/O stack
        const unsigned long nCols = std::tuple_size<Pack>::value;
        std::vector<std::vector<ClassId> > dummy;
        const ClassId& myclass = ClassId(*this);
        myclass.templateParameters(&dummy);
        assert(dummy.size() == 1U);
        dummy[0].at(0).templateParameters(&iostack_);
        assert(iostack_.size() == nCols);

        // Prepare the buffers
        readBuffers_.resize(nCols);
        for (unsigned long i=0; i<nCols; ++i)
            readBuffers_[i] = new Private::ColumnBuffer();
    }


    template <typename Pack>
    template <unsigned long N, typename Storage>
    void ColumnPacker<Pack>::fetchItem(const unsigned long row,
                                       Storage s) const
    {
        typedef typename IOReferredType<Storage>::type requested_type;
        typedef typename std::tuple_element<N,Pack>::type element_type;
        typedef typename IOReferredType<element_type>::type stored_type;

        static_assert((IOIsSameType<stored_type, "requested type>::value");
        static_assert((N < std::tuple_size<Pack>::value), "column number is out of range");

        if (row < fillCount_)
        {
            if (firstUnpack_)
            {
                prepareToUnpack();
                firstUnpack_ = false;
            }
            currentReadRow_ = row;
            std::vector<ClassId>* iost = 0;
            std::istream* is = columnIstream(N, &iost);
            if (is)
            {
                if (!process_item<GenericReader>(s, *is, iost, false))
                    throw IOInvalidData("In gs::ColumnPacker::fetchItem: "
                                        "failed to unpack tuple data");
            }
            else
                throw gs::IOInvalidArgument("In gs::ColumnPacker::fetchItem: "
                                            "invalid column number");
        }
        else
            throw gs::IOOutOfRange("In gs::ColumnPacker::fetchItem: "
                                    "row number is out of range");
    }


    template <typename Pack>
    void ColumnPacker<Pack>::rowContents(const unsigned long row,
                                         Pack* tuple) const
    {
        if (row < fillCount_)
        {
            assert(tuple);
            if (firstUnpack_)
            {
                prepareToUnpack();
                firstUnpack_ = false;
            }
            currentReadRow_ = row;
            if (!Private::ColumnPackerHelper<Pack,
                std::tuple_size<Pack>::value>::readRow(*this, tuple))
                throw IOInvalidData("In gs::ColumnPacker::rowContents: "
                                    "failed to unpack tuple data");
        }
        else
            throw gs::IOOutOfRange("In gs::ColumnPacker::rowContents: "
                                    "row number is out of range");
    }


    template <typename T>
    bool ColumnPacker<T>::write()
    {
        if (!writable_)
            return false;
        saveHeader();
        const unsigned long nBuffers = fillBuffers_.size();
        for (unsigned long i=0; i<nBuffers; ++i)
        {
            Private::ColumnBuffer& buffer(*fillBuffers_[i]);
            if (buffer.lastrowp1 > buffer.firstrow)
                saveFillBuffer(i);
        }
        ar_ << Private::CPFooterRecord<ColumnPacker<T> >(*this);
        writable_ = false;
        return true;
    }


    template <typename T>
    const char* ColumnPacker<T>::classname()
    {
        static const std::string myClass(
            template_class_name<T>("gs::ColumnPacker"));
        return myClass.c_str();
    }


    template <typename T>
    inline unsigned long ColumnPacker<T>::nColumnsReadBack() const
    {
        if (isOriginalTuple_)
            return std::tuple_size<T>::value;
        else
            return std::accumulate(readbackMask_.begin(),
                                   readbackMask_.end(), 0ULL);
    }


    template <typename T>
    unsigned long ColumnPacker<T>::columnNumber(const char* columnName) const
    {
        return findName(colNames_, columnName);
    }


    template <typename T>
    unsigned long ColumnPacker<T>::nOriginalColumns() const
    {
        if (isOriginalTuple_)
            return nColumns();
        else
            return originalColNames_->size();
    }


    template <typename T>
    const std::string& 
    ColumnPacker<T>::originalColumnName(const unsigned long i) const
    {
        if (isOriginalTuple_)
            return columnName(i);
        else
            return originalColNames_->at(i);
    }


    template <typename T>
    const std::vector<std::string>& 
    ColumnPacker<T>::originalColumnNames() const
    {
        if (isOriginalTuple_)
            return columnNames();
        else
            return *originalColNames_;
    }


    template <typename T>
    unsigned long ColumnPacker<T>::originalColumn(
        const unsigned long currentNumber) const
    {
        const unsigned long ncols = std::tuple_size<T>::value;
        if (isOriginalTuple_)
            return currentNumber < ncols ? currentNumber : ncols;
        else
        {
            if (currentNumber < ncols)
                return (*originalColNumber_)[currentNumber];
            else
                return originalColNames_->size();
        }
    }


    template <typename T>
    unsigned long ColumnPacker<T>::originalColumnNumber(
        const char* columnName) const
    {
        if (isOriginalTuple_)
            return columnNumber(columnName);
        else
            return findName(*originalColNames_, columnName);
    }


    template <typename Pack>
    template <typename Pack2>
    bool ColumnPacker<Pack>::operator==(const ColumnPacker<Pack2>& r) const
    {
        if ((void *)this == (void *)(&r))
            return true;
        if (!readable_ || !r.readable_)
            return false;

        // It is possible to compare some tuples in their "non-original"
        // form (in case they differ from the original tuple only by
        // a column permutation). However, things become significantly
        // more complicated, so this comparison is not supported.
        if (!isOriginal() || !r.isOriginal())
            return false;

        const unsigned long ncols = nColumns();
        if (ncols != r.nColumns())
            return false;
        if (fillCount_ != r.fillCount_)
            return false;
        if (title_ != r.title_)
            return false;
        if (colNames_ != r.colNames_)
            return false;

        for (unsigned long icol=0; icol<ncols; ++icol)
            for (unsigned long row=0; row<fillCount_; ++row)
            {
                unsigned long len1=0, len2=0;
                std::istream* is1 = getRowStream(row, icol, &len1);
                std::istream* is2 = r.getRowStream(row, icol, &len2);
                assert(is1);
                assert(is2);
                if (len1 != len2)
                    return false;
                std::streambuf* buf1 = is1->rdbuf();
                std::streambuf* buf2 = is2->rdbuf();
                unsigned long i=0;
                for (; i<len1 && buf1->sbumpc() == buf2->sbumpc(); ++i) {;}
                if (i < len1) return false;
            }

        return true;
    }


    template<typename Pack>
    inline const std::vector<std::string>&
    ColumnPacker<Pack>::defaultColumnNames()
    {
        return default_tuple_columns<std::tuple_size<Pack>::value>();
    }
}


#endif // CPP11_STD_AVAILABLE
#endif // GENERS_COLUMNPACKER_HH_

