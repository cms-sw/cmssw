#ifndef __class_____class___h
#define __class_____class___h
// -*- C++ -*-
/*
Class      : __class__
Created by : __author__ on __date__
Description:
Parameters :
Returns    :
Throws     :
*/

template <typename T>
class __class__ {
	// ---------- friend classes and functions ---------------
public:
	// ---------- constants, enums and typedefs --------------

	// ---------- Constructors and destructor ----------------
	__class__();
	virtual ~__class__();

	// ---------- member functions ---------------------------

	// ---------- const member functions ---------------------

	// ---------- static member functions --------------------

protected:
	// ---------- protected member functions -----------------

	// ---------- protected const member functions -----------

private:
	// ---------- Constructors and destructor ----------------
	__class__( const __class__<T>& src ); // copy-ctor
	// __class__(const __class__<T>&) = delete; // stop default
	__class__( __class__<T>&& src ); // move-ctor

	// ---------- assignment operator(s) ---------------------
	const __class__<T>& operator=( const __class__<T>& rhs ); // copy assignment oper
	// const __class__<T>& operator=(const __class__<T>&) = delete; // stop default
	__class__<T>& operator=( __class__<T>&& rhs ); // move assignment oper

	// ---------- private member functions -------------------

	// ---------- private const member functions -------------

	// ---------- data members -------------------------------

	// ---------- static data members ------------------------
};

// c++11 requires that your template implementation should be in a header file
// to do so we include here implementation file (__class__inl).
#include "__class__.inl"
#endif
