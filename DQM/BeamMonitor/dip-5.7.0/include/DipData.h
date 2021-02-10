#ifndef DIPDATA_H_INCLUDED
#define DIPDATA_H_INCLUDED

#include "DipTimestamp.h"
#include "DipQuality.h"
#include "StdTypes.h"
#include "DipDataType.h"



/**
* Object to hold either primitive or complex DIP data.
* Used on the subscriber side and may be use on the publisher side.
*/
class DipDllExp DipData
{
private:
	DipData(const DipData &);
	DipData& operator=(const DipData& other);

public:
	DipData() {}
	virtual ~DipData() {}

	/**
	* Extract string data as a std::string. The returned value is valid untill the field value
	* is over written.
	*/
	virtual const std::string &	extractString(const char *tag=NULL) const = 0;

	/**
	* Extract string data as a null terminated string. The returned value is valid untill the 
	* field value is over written. 
	* @return not owned by caller. 
	*/
	virtual const char*	extractCString(const char *tag=NULL) const = 0;

	/**
	* Extract bool.  
	*/
	virtual DipBool	extractBool(const char *tag=NULL) const = 0;

	/**
	* Extract byte. 
	*/
	virtual DipByte	extractByte(const char *tag=NULL) const = 0;

	/**
	* Extract short.  
	*/
	virtual DipShort extractShort(const char *tag=NULL) const = 0;

	/**
	* Extract int.  
	*/
	virtual DipInt extractInt(const char *tag=NULL) const = 0;

	/**
	* Extract float.  
	*/
	virtual DipFloat extractFloat(const char *tag=NULL) const = 0;

	
	/**
	* Extract double.  
	*/
	virtual DipDouble extractDouble(const char *tag=NULL) const = 0;


	/**
	* Extract long.  
	*/
	virtual DipLong	extractLong(const char *tag=NULL) const = 0;

	/**
	* Extract bool array. The returned value is valid untill the 
	* field value is over written. 
	* @return not owned by caller. 
	*/
	virtual const DipBool* extractBoolArray(int &size,	const char *tag=NULL) const = 0;

	/**
	* Extract byte array. The returned value is valid untill the 
	* field value is over written. 
	* @return not owned by caller. 
	*/
	virtual const DipByte* extractByteArray(int &size, const char *tag=NULL) const = 0;

	/**
	* Extract short array. The returned value is valid untill the 
	* field value is over written. 
	* @return not owned by caller. 
	*/
	virtual const DipShort* extractShortArray(int &size, const char *tag=NULL) const = 0;

	/**
	* Extract int array. The returned value is valid untill the 
	* field value is over written. 
	* @return not owned by caller. 
	*/
	virtual const DipInt* extractIntArray(int &size,	const char *tag=NULL) const = 0;

	/**
	* Extract float array. The returned value is valid untill the 
	* field value is over written. 
	* @return not owned by caller. 
	*/
	virtual const DipFloat* extractFloatArray(int &size, const char *tag=NULL) const = 0;

	/**
	* Extract double array. The returned value is valid untill the 
	* field value is over written. 
	* @return not owned by caller. 
	*/
	virtual const DipDouble* extractDoubleArray(int &size, const char *tag=NULL) const = 0;

	/**
	* Extract long array. The returned value is valid untill the 
	* field value is over written. 
	* @return not owned by caller. 
	*/
	virtual const DipLong* extractLongArray(int &size,	const char *tag=NULL) const = 0;

	/**
	* Extract string array data as an array std::string. The returned value is valid untill the 
	* field value is over written. 
	* @return not owned by caller. 
	*/
	virtual const std::string* extractStringArray(int &size, const char *tag=NULL) const = 0;

	/**
	* Extract string array data as a null terminated string array. The returned value is valid untill the 
	* field value is over written. 
	* @return not owned by caller. 
	*/
	virtual const char** extractCStringArray(int &size,const char *tag=NULL) const = 0;

	/**
	* Insert a string into the object as a primitive/member of a structure
	* @param tag if NULL then the object is primitive, ownedship remains with caller.
	*/
	virtual void insert(const std::string &value,const char *tag=NULL) = 0;

	/**
	* Insert a string into the object as a primitive/member of a structure
	* @param tag if NULL then the object is primitive, ownedship remains with caller.
	*/
	virtual void insert(const char value[], const char *tag=NULL) = 0;

	/**
	* Insert a bool into the object as a primitive/member of a structure
	* @param tag if NULL then the object is primitive, ownedship remains with caller.
	*/
	virtual void insert(DipBool value, const char *tag=NULL) = 0;

	/**
	* Insert a byte into the object as a primitive/member of a structure
	* @param tag if NULL then the object is primitive, ownedship remains with caller.
	*/
	virtual void insert(DipByte value, const char *tag=NULL) = 0;

	/**
	* Insert a short into the object as a primitive/member of a structure
	* @param tag if NULL then the object is primitive, ownedship remains with caller.
	*/
	virtual void insert(DipShort value, const char *tag=NULL) = 0;

	/**
	* Insert an int into the object as a primitive/member of a structure
	* @param tag if NULL then the object is primitive, ownedship remains with caller.
	*/
	virtual void insert(DipInt value, const char *tag=NULL) = 0;

	/**
	* Insert a long into the object as a primitive/member of a structure
	* @param tag if NULL then the object is primitive, ownedship remains with caller.
	*/
	virtual void insert(DipLong value, const char *tag=NULL) = 0;

	/**
	* Insert a float into the object as a primitive/member of a structure
	* @param tag if NULL then the object is primitive, ownedship remains with caller.
	*/
	virtual void insert(DipFloat value,	const char *tag=NULL) = 0;

	/**
	* Insert a double into the object as a primitive/member of a structure
	* @param tag if NULL then the object is primitive, ownedship remains with caller.
	*/
	virtual void insert(DipDouble value, const char * tag=NULL) = 0;

	/**
	* Insert a bool array into the object as a primitive/member of a structure
	* @param addr array containing the value to be entered into the object. Copied.
	* @param tag if NULL then the object is primitive, ownedship remains with caller.
	*/
	virtual void insert(const DipBool addr[], int size, const char *tag=NULL) = 0;

	/**
	* Insert a byte array into the object as a primitive/member of a structure
	* @param addr array containing the value to be entered into the object. Copied.
	* @param tag if NULL then the object is primitive, ownedship remains with caller.
	*/
	virtual void insert(const DipByte addr[], int size, const char *tag=NULL) = 0;

	/**
	* Insert a short array into the object as a primitive/member of a structure
	* @param addr array containing the value to be entered into the object. Copied.
	* @param tag if NULL then the object is primitive, ownedship remains with caller.
	*/
	virtual void insert(const DipShort addr[], int size, const char *tag=NULL) = 0;

	/**
	* Insert a int array into the object as a primitive/member of a structure
	* @param addr array containing the value to be entered into the object. Copied.
	* @param tag if NULL then the object is primitive, ownedship remains with caller.
	*/
	virtual void insert(const DipInt addr[], int size, const char *tag=NULL) = 0;

	/**
	* Insert a long array into the object as a primitive/member of a structure
	* @param addr array containing the value to be entered into the object. Copied.
	* @param tag if NULL then the object is primitive, ownedship remains with caller.
	*/
	virtual void insert(const DipLong addr[], int size, const char *tag=NULL) = 0;

	/**
	* Insert a float array into the object as a primitive/member of a structure
	* @param addr array containing the value to be entered into the object. Copied.
	* @param tag if NULL then the object is primitive, ownedship remains with caller.
	*/
	virtual void insert(const DipFloat addr[], int size, const char *tag=NULL) = 0;

	/**
	* Insert a double array into the object as a primitive/member of a structure
	* @param addr array containing the value to be entered into the object. Copied.
	* @param tag if NULL then the object is primitive, ownedship remains with caller.
	*/
	virtual void insert(const DipDouble addr[],	int size, const char *tag=NULL) = 0;

	/**
	* Insert a string array into the object as a primitive/member of a structure
	* @param addr array containing the value to be entered into the object. Copied.
	* @param tag if NULL then the object is primitive, ownedship remains with caller.
	*/
	virtual void insert(const std::string value[],int size,const char *tag=NULL) = 0;

	/**
	* Insert a string array into the object as a primitive/member of a structure
	* @param addr array containing the value to be entered into the object. Copied.
	* @param tag if NULL then the object is primitive, ownedship remains with caller.
	*/
	virtual void insert(const char *value[], int size, const char *tag=NULL) = 0;


	/**
	* if called with tag=NULL then the method will return the time of the default
	* field (primitive). If the object is complex then the return value will be TYPE_NULL.
	* if called with tag!=NULL then the field represented by the tag name passed
	* is returned. If the field does not exist then TYPE_NULL is returned 
    * @see #TYPE_NULL
    * @see #TYPE_BOOLEAN
    * @see #TYPE_BOOLEAN_ARRAY
    * @see #TYPE_BYTE
    * @see #TYPE_BYTE_ARRAY
    * @see #TYPE_SHORT
    * @see #TYPE_SHORT_ARRAY
    * @see #TYPE_INT
    * @see #TYPE_INT_ARRAY
    * @see #TYPE_LONG
    * @see #TYPE_LONG_ARRAY
    * @see #TYPE_FLOAT
    * @see #TYPE_FLOAT_ARRAY
    * @see #TYPE_DOUBLE
    * @see #TYPE_DOUBLE_ARRAY
    * @see #TYPE_STRING
    * @see #TYPE_STRING_ARRAY
    */
	virtual DipDataType getValueType(const char *tag=NULL) const = 0;

	/**
	* Returns the CURRENT fields that form this Data Object
	* Returned array of tag names is NOT owned by the caller - it is valid as
	* long as no extra fields are added to the DataObject. if extra fields are
	* added last returned ptr will be invalid when this method is called again.
	* If no fields are added an empty array will be returned.
	*/
	virtual const char ** getTags(int &nTags) const = 0;

	/**
	* Used to determine if a tag (field) is contained in this object
	* @param tag field to look for
	* @return true if tag exists, false otherwise
	*/
	virtual bool contains(const char tag[]) const = 0;

	/**
	* Number of fields in this object
	*/
	virtual unsigned int size() const = 0;


	/**
	*
	*/
	virtual bool isEmpty() const= 0;

	/**
	* Provides information of the 'trust worthiness of the data' in the data object
	* @see DipQuality
	*/
	virtual DipQuality	extractDataQuality() const = 0;

	/**
	* provides supplemental information (if provided by the server) on the data quality when the quality is 
	* not DIP_QUALITY_GOOD
	* @return empty string if quality is good or information not provided by the publisher, otherwise publisher specific string
	*/
	virtual const std::string & extractReasonForQuality() const = 0;

	/**
	* Retrive the time object that describes when the data value held in this object
	* was retrieved.
	*/
	virtual const DipTimestamp & extractDipTime() const = 0;

	/**
	* Get the dimension >= 1 of the supplied field, 0 if field
	* does not exist
	*/
	virtual unsigned getValueDimension(const char *tag) const = 0;
};



// Making the DipData printable to an ostream
std::ostream& operator<<(std::ostream& theStream, DipData& theData);	// TODO // implementation





#endif // DIPDATA_H_INCLUDED
