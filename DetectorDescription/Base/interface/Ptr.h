#ifndef GUARD_Ptr_h
#define GUARD_Ptr_h

#include <stdexcept>
#include <cstddef>
#include <iostream>
#include <utility>

template<class T> class Ptr {
public:
	// new member to copy the object conditionally when needed
	void make_unique() {
		if (*refptr != 1) {
			--*refptr;
			refptr = new size_t(1);
			p = p? clone(p): 0;
		}
	}

	// the rest of the class looks like `Ref_handle' except for its name
	Ptr(): p(0), refptr(new size_t(1))  { }
	Ptr(T* t): p(t), refptr(new size_t(1)) { }
	Ptr(const Ptr& h): p(h.p), refptr(h.refptr)  { ++*refptr; }

	Ptr& operator=(const Ptr&);    // implemented analogously to 14.2/261
	~Ptr();                        // implemented analogously to 14.2/262
	operator bool() const { return p; }
	T& operator*() const;          // implemented analogously to 14.2/261
	T* operator->() const;         // implemented analogously to 14.2/261
	
	//FIXME: Ptr::operator==()is this good design?
	bool operator==( const Ptr<T> P) const { return p==P.p; }
        
	//FIXME: Ptr::operator<()is this good design?
	bool operator<( const Ptr<T> P) const { return p<P.p; }
	  
private:
	T* p;
	std::size_t* refptr;
};

template<class T> T* clone(const T* tp)
{
	return tclone(tp);
}



template<class T>
T& Ptr<T>::operator*() const { if (p) return *p; throw std::runtime_error("unbound Ptr"); }

template<class T>
T* Ptr<T>::operator->() const { if (p) return p; throw std::runtime_error("unbound Ptr"); }


template<class T>
Ptr<T>& Ptr<T>::operator=(const Ptr& rhs)
{
        ++*rhs.refptr;
        // free the lhs, destroying pointers if appropriate
        if (--*refptr == 0) {
                delete refptr;
                delete p;
        }

        // copy in values from the right-hand side
        refptr = rhs.refptr;
        p = rhs.p;
        return *this;
}

template<class T> Ptr<T>::~Ptr()
{
        if (--*refptr == 0) {
                delete refptr;
/*
		if (p)
		  std::cout << "DC:\t\tPtr deleted: " << std::endl;
		else
		  std::cout << "DC: \t\tPtr deleted: dangling" << std::endl;  
                delete p;
*/		
        }
}


#endif
