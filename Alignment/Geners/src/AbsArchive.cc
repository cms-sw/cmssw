#include <cassert>
#include <sstream>

#include "Alignment/Geners/interface/AbsArchive.hh"
#include "Alignment/Geners/interface/IOException.hh"

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
    long long delta = off;
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
