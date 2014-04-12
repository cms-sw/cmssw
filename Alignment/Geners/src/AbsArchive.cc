#include <cassert>
#include <sstream>

#include "Alignment/Geners/interface/AbsArchive.hh"
#include "Alignment/Geners/interface/IOException.hh"

#define GS_STREAM_COPY_BUFFER_SIZE 65536

static void archive_stream_copy(std::istream &in, std::size_t count,
                                std::ostream &out)
{
    if (count)
    {
        const std::size_t bufsize = GS_STREAM_COPY_BUFFER_SIZE;
        char buffer[bufsize];

        bool in_fail = in.fail();
        bool out_fail = out.fail();
        while (count > bufsize && !in_fail && !out_fail)
        {
            in.read(buffer, bufsize);
            in_fail = in.fail();
            if (!in_fail)
            {
                out.write(buffer, bufsize);
                out_fail = out.fail();
            }
            count -= bufsize;
        }
        if (!in_fail && !out_fail)
        {
            in.read(buffer, count);
            if (!in.fail())
                out.write(buffer, count);
        }
    }
}

namespace {
    class NotWritableRecord : public gs::AbsRecord
    {
    public:
        inline NotWritableRecord(const gs::ClassId& classId,
                                 const char* ioPrototype,
                                 const char* name, const char* category)
            : gs::AbsRecord(classId, ioPrototype, name, category) {}
    private:
        NotWritableRecord();
        inline bool writeData(std::ostream&) const {return false;}
    };
}

namespace gs {
    AbsArchive::AbsArchive(const char* name)
        : name_(name ? name : ""),
          lastItemId_(0),
          lastItemLength_(0)
    {
    }

    void AbsArchive::addItemToReference(AbsReference& r,
                                        const unsigned long long id) const
    {
        r.addItemId(id);
    }

    unsigned long long AbsArchive::copyItem(const unsigned long long id,
                                            AbsArchive* destination,
                                            const char* newName,
                                            const char* newCategory)
    {
        if (!isReadable())
            throw gs::IOInvalidArgument("In gs::AbsArchive::copyItem: "
                                        "origin archive is not readable");
        assert(destination);
        if (this == destination)
            throw gs::IOInvalidArgument("In gs::AbsArchive::copyItem: "
                         "origin and destination archives are the same");
        AbsArchive& ar(*destination);
        if (!ar.isWritable())
            throw gs::IOInvalidArgument("In gs::AbsArchive::copyItem: "
                                        "destination archive is not writable");
        CPP11_shared_ptr<const CatalogEntry> entry(catalogEntry(id));
        if (!entry)
            throw gs::IOInvalidArgument("In gs::AbsArchive::copyItem: no item "
                                        "in the archive with the given id");

        // Position the input stream
        long long sz = 0;
        std::istream& is = inputStream(id, &sz);

        // The following code is similar to the one in the "operator<<"
        // below w.r.t. operations with the output archive
        std::ostream& os = ar.outputStream();
        std::streampos base = os.tellp();
        std::ostream& compressed = ar.compressedStream(os);

        // Transfer the data between the streams
        unsigned long long len;
        if (sz >= 0)
            len = sz;
        else
            len = entry->itemLength();
        archive_stream_copy(is, len, compressed);
        if (is.fail())
            throw IOReadFailure("In gs::AbsArchive::copyItem: "
                                "input stream failure");
        if (compressed.fail())
            throw IOWriteFailure("In gs::AbsArchive::copyItem: "
                                 "output stream failure");
        const unsigned compressCode = ar.flushCompressedRecord(compressed);
        if (os.fail())
            throw IOWriteFailure("In gs::AbsArchive::copyItem: "
                                 "failed to transfer compressed data");

        // Figure out the record length. Naturally, can't have negative length.
        std::streamoff off = os.tellp() - base;
        const long long delta = off;
        assert(delta >= 0LL);

        // We need to create a record out of the catalog entry we have found
        const char* name = newName;
        if (!name)
            name = entry->name().c_str();
        const char* category = newCategory;
        if (!category)
            category = entry->category().c_str();
        NotWritableRecord record(entry->type(), entry->ioPrototype().c_str(),
                                 name, category);

        // Add the record to the catalog
        const unsigned long long id2 = ar.addToCatalog(
            record, compressCode, delta);
        if (id == 0ULL)
            throw IOWriteFailure("In gs::AbsArchive::copyItem: "
                                 "failed to add catalog entry");
        ar.lastItemId_ = id2;
        ar.lastItemLength_ = delta;
        return id2;
    }
}

static std::string local_error_message(gs::AbsArchive& ar,
                                       const gs::AbsRecord& record,
                                       const char* failedAction)
{
    std::ostringstream err;
    err << "In operator<<(gs::AbsArchive& ar, const gs::AbsRecord& record): "
        << "failed to " << failedAction << " to the archive \""
        << ar.name() << "\" for item with type \""
        << record.type().name() << "\", name \""
        << record.name() << "\", and category \""
        << record.category() << '"';
    return err.str();
}

gs::AbsArchive& operator<<(gs::AbsArchive& ar, const gs::AbsRecord& record)
{
    // Do not reuse records
    if (record.id()) throw gs::IOInvalidArgument(
        "In operator<<(gs::AbsArchive& ar, const gs::AbsRecord& record): "
        "records can not be reused");

    // Get the relevant streams. Do not change the order
    // of the next three lines, some code which does
    // not implement compression may actually rely on
    // the fact that the "outputStream()" method
    // is called before "compressedStream()".
    std::ostream& os = ar.outputStream();
    std::streampos base = os.tellp();
    std::ostream& compressed = ar.compressedStream(os);

    // Write the data
    if (!record.writeData(compressed))
        throw gs::IOWriteFailure(local_error_message(ar, record, "write data"));

    const unsigned compressCode = ar.flushCompressedRecord(compressed);
    if (os.fail())
        throw gs::IOWriteFailure(local_error_message(
                                     ar, record, "transfer compressed data"));

    // Figure out the record length. Naturally, can't have negative length.
    std::streamoff off = os.tellp() - base;
    const long long delta = off;
    assert(delta >= 0LL);

    // Add the record to the catalog
    const unsigned long long id = ar.addToCatalog(record, compressCode, delta);
    if (id == 0ULL)
        throw gs::IOWriteFailure(local_error_message(
                                     ar, record, "add catalog entry"));

    // Mark record as written and give feedback about item length
    record.itemId_ = id;
    record.itemLength_ = delta;

    // Same thing for the archive
    ar.lastItemId_ = id;
    ar.lastItemLength_ = delta;

    return ar;
}
