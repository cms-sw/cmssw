/*
* Copyright 1993-2015 NVIDIA Corporation.  All rights reserved.
*
* Please refer to the NVIDIA end user license agreement (EULA) associated
* with this source code for terms and conditions that govern your use of
* this software. Any use, reproduction, disclosure, or distribution of
* this software and related documentation outside the terms of the EULA
* is strictly prohibited.
*
*/


/////////////////////////////////////////////////////////////////////////////
//
// Container parent class.
//
////////////////////////////////////////////////////////////////////////////


template<class T>
class Container {

public:
	__device__
	Container() {;}

       __device__
        virtual ~Container() {;}

	__device__
	virtual void push(T e) = 0;

	__device__
	virtual bool pop(T &e) = 0;
};

/////////////////////////////////////////////////////////////////////////////
//
// Vector class derived from Container class using linear memory as data storage
// NOTE: This education purpose implementation has restricted functionality. 
//       For example, concurrent push and pop operations will not work correctly.
//
////////////////////////////////////////////////////////////////////////////


template<class T>
class Vector : public Container<T> {

public:
	// Constructor, data is allocated on the heap
    // NOTE: This must be called from only one thread
	__device__
	Vector(int max_size) :  m_top(-1) {
		m_data = new T[max_size];
	}

	// Constructor, data uses preallocated buffer via placement new
	__device__
	Vector(int max_size, T* preallocated_buffer) :  m_top(-1) {
		m_data = new (preallocated_buffer) T[max_size];
	}

    // Destructor, data is freed 
    // NOTE: This must be called from only one thread
	__device__
	~Vector() {
		if( m_data ) delete [] m_data;
	}

	__device__
	virtual
	void push(T e) {
        if( m_data ) {
		    // Atomically increment the top idx
		    int idx = atomicAdd(&(this->m_top), 1);
		    m_data[idx+1] = e;
        }
	}

	__device__
	virtual
	bool pop(T &e) {
		if( m_data && m_top >= 0 ) {
			// Atomically decrement the top idx
			int idx = atomicAdd( &(this->m_top), -1 );
			if( idx >= 0 ) {
				e = m_data[idx];
				return true;
			}
		}
		return false;
		
	}


private:
	int m_size;
	T* m_data;

	int m_top;
};
