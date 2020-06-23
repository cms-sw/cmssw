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
	__class__( const __class__& src ); // copy-ctor
	// __class__(const __class__&) = delete; // stop default
	__class__( __class__&& src ); // move-ctor

	// ---------- assignment operator(s) ---------------------
	const __class__& operator=( const __class__& rhs ); // copy assignment oper
	// const __class__& operator=(const __class__&) = delete; // stop default
	__class__& operator=( __class__&& rhs ); // move assignment oper

	// ---------- private member functions -------------------

	// ---------- private const member functions -------------

	// ---------- data members -------------------------------

	// ---------- static data members ------------------------
};
#endif
