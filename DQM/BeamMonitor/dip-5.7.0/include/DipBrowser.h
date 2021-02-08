#ifndef DIPBROWSER_H
#define DIPBROWSER_H

#include "DipException.h"

/**
 * Interface for DIP Browser Class
 */
class DipDllExp DipBrowser
{	
	public:
	/**
	* @param pattern - wild card filter - may be null if no filter is to be applied 
	* @param noPublications - number of publication names returned
	* @return list of DIP services matching the wild card pattern - this is NOT owned by caller. Valid until next call.
	*/
	virtual const char ** getPublications(const char * pattern, unsigned int &noPublications) = 0;

	/**
	 * Get the fields of publication of the supplied name. This may be an emty list if the
	 * publication has not yet published any data (and thus the field structure is undefined).
	 * A primitive type will return the tag name __DIP_DEFAULT__
	 * @param pub publication who's field names we are to try to get.
	 * @param noField the number of field names retreived for a field.
	 * @return list of publication fields - this is NOT owned by caller. Valid until next call. 
	 * @throws DipException if the publication does not exist
	 * */
	virtual const char ** getTags(const char * pub, unsigned int &noFields) = 0;
	
	/**
     * Get the type of the default field (where a primitive type is being sent).
	 * Must be called after first calling getTags() for the publication of interest.
     * @return type of the primitive sent in the DipData object as defined in the DipData class
     * @throws DipException if getTags() has not been previous called or the object does not hold a primitive type
     * @see DipBrowser.getTags().
     * @see DipData
	 * */
	virtual int getType() = 0;

	/**
     * Get the cardinality of the default field (where a primitive type is being sent).
	 * Must be called after first calling getTags() for the publication of interest.
	 * Note that a non array type will return the value 1.
     * @return cardinality of the primitive sent in the DipData object.
     * @throws DipException if getTags() has not been previous called or the object does not hold a primitive type
     * @see DipBrowser.getTags().
     * @see DipData
	 * */
	virtual int getSize() = 0;

	/**
     * Get the type of the field who's name is supplied as a parameter.
	 * Must be called after first calling getTags() for the publication of interest.
     * @param name of the field whos type we are interested in.
     * @return type of the field sent in the DipData object as defined in the DipData class
     * @throws DipException if getTags() has not been previous called or the object does not hold a complex type
     * @see DipBrowser.getTags().
     * @see DipData
	 * */
	virtual int getType(const char * tag) = 0;

	 /**
     * Get the cardinality of the field who's name is supplied as a parameter.
	 * Must be called after first calling getTags() for the publication of interest.
     * Note that a non array type will return the value 1.
     * @param name of the field whos cardinality we are interested in.
     * @return cardinality of the field sent in the DipData object as defined in the DipData class
     * @throws DipException if getTags() has not been previous called or the object does not hold a complex type
     * @see DipBrowser.getTags().
     * @see DipData
	 * */
	virtual int getSize(const char * tag) = 0;

	virtual ~DipBrowser() = 0;
};


#endif
