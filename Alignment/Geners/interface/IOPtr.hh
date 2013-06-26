#ifndef GENERS_IOPTR_HH_
#define GENERS_IOPTR_HH_

#include <string>
#include "Alignment/Geners/interface/IOException.hh"

// "Silent" pointers for the I/O system
namespace gs {
    template <typename T> class IOProxy;

    template <typename T>
    class IOPtr
    {
        template <typename T2> friend class IOProxy;

    public:
        typedef T element_type;

        inline IOPtr() : ptr_(0) {}
        inline IOPtr(T* ptr) : ptr_(ptr) {}
        IOPtr(const IOProxy<T>& p);

        IOPtr& operator=(const IOProxy<T>& p);
        inline IOPtr& operator=(const IOPtr& p)
            {ptr_ = p.ptr_; return *this;}

        // Typical pointer operations
        inline T* get() const {return ptr_;}
        inline T* operator->() const
        {
            if (!ptr_) throw gs::IOInvalidArgument(
               "In gs::IOPtr::operator->: attempt to dereference null pointer");
            return ptr_;
        }
        inline T& operator*() const
        {
            if (!ptr_) throw gs::IOInvalidArgument(
               "In gs::IOPtr::operator*: attempt to dereference null pointer");
            return *ptr_;
        }
        inline IOPtr& operator++() {++ptr_; return *this;}
        inline void operator++(int) {++ptr_;}
        inline IOPtr& operator--() {--ptr_; return *this;}
        inline void operator--(int) {--ptr_;}

        // Implicit conversion to "bool"
        inline operator bool() const {return !(ptr_ == 0);}

        // Names are ignored during comparisons
        inline bool operator==(const IOPtr& r) const {return ptr_ == r.ptr_;}
        inline bool operator!=(const IOPtr& r) const {return ptr_ != r.ptr_;}
        bool operator==(const IOProxy<T>& r) const;
        bool operator!=(const IOProxy<T>& r) const;

        // Should be able to work as with bare pointer inside the I/O code
        inline T*& getIOReference() {return ptr_;}
        inline T* const & getIOReference() const
        {
            if (!ptr_) throw gs::IOInvalidArgument(
                "In gs::IOPtr::getIOReference: unusable "
                "const reference to null pointer");
            return ptr_;
        }

    private:
        T* ptr_;
    };

    template <typename T>
    class IOProxy
    {
        template <typename T2> friend class IOPtr;

    public:
        typedef T element_type;

        inline IOProxy() : ptr_(0) {}
        inline IOProxy(T* ptr) : ptr_(ptr) {}
        inline IOProxy(T* ptr, const char* varname)
            : ptr_(ptr), name_(varname ? varname : "") {}
        inline IOProxy(T* ptr, const std::string& varname)
            : ptr_(ptr), name_(varname) {}
        inline IOProxy(const IOPtr<T>& p) : ptr_(p.ptr_) {}
        inline IOProxy(const IOPtr<T>& p, const char* varname)
            : ptr_(p.ptr_), name_(varname ? varname : "") {}
        inline IOProxy(const IOPtr<T>& p, const std::string& varname)
            : ptr_(p.ptr_), name_(varname) {}

        inline IOProxy& operator=(const IOProxy& p)
            {ptr_ = p.ptr_; name_ = p.name_; return *this;}
        inline IOProxy& operator=(const IOPtr<T>& p)
            {ptr_ = p.ptr_; name_ = ""; return *this;}

        // Get name
        inline const std::string& name() const {return name_;}

        // Set name
        inline IOProxy& setName(const char* varname)
        {
            name_ = varname ? varname : "";
            return *this;
        }
        inline IOProxy& setName(const std::string& varname)
        {
            name_ = varname;
            return *this;
        }

        // Typical pointer operations
        inline T* get() const {return ptr_;}
        inline T* operator->() const
        {
            if (!ptr_) throw gs::IOInvalidArgument(
               "In gs::IOProxy::operator->: attempt to dereference null pointer");
            return ptr_;
        }
        inline T& operator*() const
        {
            if (!ptr_) throw gs::IOInvalidArgument(
               "In gs::IOProxy::operator*: attempt to dereference null pointer");
            return *ptr_;
        }
        inline IOProxy& operator++() {++ptr_; return *this;}
        inline void operator++(int) {++ptr_;}
        inline IOProxy& operator--() {--ptr_; return *this;}
        inline void operator--(int) {--ptr_;}

        // Implicit conversion to "bool"
        inline operator bool() const {return !(ptr_ == 0);}

        // Names are ignored during comparisons
        inline bool operator==(const IOProxy& r) const {return ptr_ == r.ptr_;}
        inline bool operator!=(const IOProxy& r) const {return ptr_ != r.ptr_;}
        inline bool operator==(const IOPtr<T>& r) const {return ptr_== r.ptr_;}
        inline bool operator!=(const IOPtr<T>& r) const {return ptr_!= r.ptr_;}

        // Should be able to work as with bare pointer inside the I/O code
        inline T*& getIOReference() {return ptr_;}
        inline T* const & getIOReference() const 
        {
            if (!ptr_) throw gs::IOInvalidArgument(
                "In gs::IOProxy::getIOReference: unusable "
                "const reference to null pointer");
            return ptr_;
        }

    private:
        T* ptr_;
        std::string name_;
    };

    // Convenience function for making IOPtr objects
    template <typename T>
    inline IOPtr<T> make_IOPtr(T& obj)
    {
        return IOPtr<T>(&obj);
    }

    // In the user code, the following function can be usually wrapped
    // as follows:
    //
    // #define io_proxy(obj) gs::make_IOProxy( obj , #obj )
    //
    template <typename T>
    inline IOProxy<T> make_IOProxy(T& obj, const char* name)
    {
        return IOProxy<T>(&obj, name);
    }
}

// IOPtr methods which could not be defined earlier
namespace gs {
    template <typename T>
    inline IOPtr<T>::IOPtr(const IOProxy<T>& p) : ptr_(p.ptr_) {}

    template <typename T>
    inline IOPtr<T>& IOPtr<T>::operator=(const IOProxy<T>& p)
    {ptr_ = p.ptr_; return *this;}

    template <typename T>
    inline bool IOPtr<T>::operator==(const IOProxy<T>& r) const 
    {return ptr_ == r.ptr_;}

    template <typename T>
    inline bool IOPtr<T>::operator!=(const IOProxy<T>& r) const 
    {return ptr_ != r.ptr_;}
}

#endif // GENERS_IOPTR_HH_

