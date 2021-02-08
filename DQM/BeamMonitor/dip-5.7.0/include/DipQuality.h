#ifndef DIPQUALITY_H_INCLUDED
#define DIPQUALITY_H_INCLUDED

enum DipQuality {
	/**
	* Inidcates the DataObject contains uninitialised data.
	*/
	DIP_QUALITY_UNINITIALIZED = -1,
	/**
	* Indicates that data can not be used. This only occurs when the publication is uninitialised.
	*/
	DIP_QUALITY_BAD = 0,
	/**
	* Indicates that data can definitely be trusted. The last update attempt by the publisher was 
	* successfull and the value(s) that this quality corresponds to are the most current available.
	*/
	DIP_QUALITY_GOOD = 1,
	/**
	* Indicates that the last update attempt by the publisher had failed (the publication data source was
	* not accessible). The value(s) this quality corresponds to is the last known good value - however, the 
	* value can no longer be considered up-to-date.
	*/
	DIP_QUALITY_UNCERTAIN = 2
};

#endif //DIPQUALITY_H_INCLUDED
