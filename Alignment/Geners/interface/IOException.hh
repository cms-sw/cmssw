#ifndef GENERS_IOEXCEPTION_HH_
#define GENERS_IOEXCEPTION_HH_

#include <string>

#include "FWCore/Utilities/interface/Exception.h"

namespace gs {
    /** Base class for the exceptions specific to the Geners I/O library */
    struct IOException : public cms::Exception
    {
        inline IOException() : cms::Exception("gs::IOException") {}

        inline explicit IOException(const std::string& description)
            : cms::Exception(description) {}

        inline explicit IOException(const char* description)
            : cms::Exception(description) {}

        virtual ~IOException() throw() {}
    };

    struct IOLengthError : public IOException
    {
        inline IOLengthError() : IOException("gs::IOLengthError") {}

        inline explicit IOLengthError(const std::string& description)
            : IOException(description) {}

        virtual ~IOLengthError() throw() {}
    };

    struct IOOutOfRange : public IOException
    {
        inline IOOutOfRange() : IOException("gs::IOOutOfRange") {}

        inline explicit IOOutOfRange(const std::string& description)
            : IOException(description) {}

        virtual ~IOOutOfRange() throw() {}
    };

    struct IOInvalidArgument : public IOException
    {
        inline IOInvalidArgument() : IOException("gs::IOInvalidArgument") {}

        inline explicit IOInvalidArgument(const std::string& description)
            : IOException(description) {}

        virtual ~IOInvalidArgument() throw() {}
    };

    /* Automatic replacement end} */

    /**
    // Throw this exception to indicate failure of various stream
    // opening methods if it is difficult or impossible to clean up
    // after the failure in the normal flow of control
    */
    class IOOpeningFailure : public IOException
    {
        inline static std::string fileOpeningFailure(
            const std::string& whereInTheCode,
            const std::string& filename)
        {
            std::string msg("In ");
            msg += whereInTheCode;
            msg += ": failed to open file \"";
            msg += filename;
            msg += "\"";
            return msg;
        }

    public:
        inline IOOpeningFailure() : IOException("gs::IOOpeningFailure") {}

        inline explicit IOOpeningFailure(const std::string& description)
            : IOException(description) {}

        inline IOOpeningFailure(const std::string& whereInTheCode,
                                const std::string& filename)
            : IOException(fileOpeningFailure(whereInTheCode, filename)) {}

        virtual ~IOOpeningFailure() throw() {}
    };

    /**
    // Throw this exception to indicate failure in the writing process.
    // For example, fail() method of the output stream returns "true",
    // and the function is unable to handle this situation locally.
    */
    struct IOWriteFailure : public IOException
    {
        inline IOWriteFailure() : IOException("gs::IOWriteFailure") {}

        inline explicit IOWriteFailure(const std::string& description)
            : IOException(description) {}

        virtual ~IOWriteFailure() throw() {}
    };

    /**
    // Throw this exception to indicate failure in the reading process.
    // For example, fail() method of the input stream returns "true",
    // and the function is unable to handle this situation locally.
    */
    struct IOReadFailure : public IOException
    {
        inline IOReadFailure() : IOException("gs::IOReadFailure") {}

        inline explicit IOReadFailure(const std::string& description)
            : IOException(description) {}

        virtual ~IOReadFailure() throw() {}
    };

    /**
    // Throw this exception when improperly formatted or invalid data
    // is detected
    */
    struct IOInvalidData : public IOException
    {
        inline IOInvalidData() : IOException("gs::IOInvalidData") {}

        inline explicit IOInvalidData(const std::string& description)
            : IOException(description) {}

        virtual ~IOInvalidData() throw() {}
    };
}

#endif // GENERS_IOEXCEPTION_HH_

