#ifndef datablock_h
#define datablock_h

#include "platformDependantOptions.h"
//#define PLATFORMDEPENDANT_DLL_API
#include <stdexcept>

/**
* Used to serialise/deserialse data
*/
class PLATFORMDEPENDANT_DLL_API DataBlock{
private:
	/**
	* Continious block of data
	*/
	char * dataHolder;

	/// Size of data holder
	unsigned int dataHolderByteSize;

	DataBlock(const DataBlock& p);
	const DataBlock & operator=(const DataBlock& p);

	/// Points 1 byte past the last byte of the last data read/written
	unsigned int currentOffset;

	/// Counts number of bytes stored
	unsigned int numBytesStored;

public:
	/**
	* Copy noBytes of data into DataBlock
	* @param data - OWNERSHIP remains with caller
	*/
	DataBlock(const void * data, unsigned int noBytes);

	/**
	* @param noBytes Number of bytes to reserve
	*/
	DataBlock(unsigned int noBytes);

	virtual ~DataBlock();


	unsigned int getNumBytesStored(){
		return numBytesStored;
	}



	void * getDataBlock() const {
		return 	dataHolder;
	}



	/**
	* Write noBytes from data into current position
	* Must be to consec. bytes.
	* @param data - OWNERSHIP remains with caller
	*/
	void write(const void * data, const unsigned int noBytes);

	/**
	* Write noBytes from data into startOffset_byte
	* must be to consec bytes
	* @param data - OWNERSHIP remains with caller
	* @param startOffset_byte - starting from 0
	*/
	void write(const void * data, const unsigned int noBytes, const unsigned int startOffset_byte);

	/**
	* Read noBytes from current position placing results in data
	* @returns ptr to data - not owned
	*/
	const void * read(const unsigned int noBytes);

	/**
	* Read noBytes from startOffset_byte placing results in data
	* @param startOffset_byte - starting from 0
	* @returns ptr to data - not owned
	*/
	const void * read(const unsigned int noBytes, const unsigned int startOffset_byte);


	/**
	* Read String (null terminated) starting from current position
	* @returns ptr to string - not owned
	*/
	const char * readString();

	/**
	* Read String (null terminated) starting from startOffset_byte
	* @returns ptr to string - not owned
	*/
	const char * readString(const unsigned int startOffset_byte);

	/*
	* @returns ptr to string - not owned
	*/
	static const char * readString(const void * data, const unsigned int dataByteSize, const unsigned int startOffset_byte);
};

#endif

