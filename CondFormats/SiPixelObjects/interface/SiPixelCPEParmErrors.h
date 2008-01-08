#ifndef CondFormats_SiPixelCPEParmErrors_h
#define CondFormats_SiPixelCPEParmErrors_h 1

#include <vector>

//--- Maybe should make this a class const, but that'd be too much work.
//--- This usage is not worth it, since in the debugger will be obvious
//--- what this is! ;)
#define NONSENSE -99999.9
#define NONSENSE_I -99999

class SiPixelCPEParmErrors {
 public:
	//! A struct to hold information for a given (alpha,beta,size)
	struct DbEntry {
		float sigma;
		float rms;
		float bias;       // For irradiated pixels
		float pix_height; // For decapitation
		float ave_Qclus;  // Average cluster charge, For 
	  DbEntry() : sigma(NONSENSE), rms(NONSENSE), 
				 bias(NONSENSE), pix_height(NONSENSE), 
				 ave_Qclus(NONSENSE) {}
	  ~DbEntry() {}
	};
	typedef std::vector<DbEntry> DbVector;

	//! A struct to hold the binning information for (part, size, alpha, beta)
	struct DbEntryBinSize {
		int partBin_size;
		int sizeBin_size;
		int alphaBin_size;
		int betaBin_size;
		DbEntryBinSize() : partBin_size(NONSENSE_I), sizeBin_size(NONSENSE_I),
				           alphaBin_size(NONSENSE_I), betaBin_size(NONSENSE_I) {}
		~DbEntryBinSize() {}
	};
	typedef std::vector<DbEntryBinSize> DbBinSizeVector;

	SiPixelCPEParmErrors() : errors_(), errorsBinSize_() {}
	virtual ~SiPixelCPEParmErrors(){}

	//!  Accessors for the vectors -- non-const version
	inline DbVector & errors() { return errors_ ; }
	inline DbBinSizeVector & errorsBin() { return errorsBinSize_ ; }

	//!  Accessors for the vectors -- const version
	inline const DbVector & errors() const { return errors_ ; }
	inline const DbBinSizeVector & errorsBinSize() const { return errorsBinSize_ ; }

	//!  Reserve some reasonable sizes for the vectors. 
	inline void reserve() {
	  errors_.reserve(1000);
		errorsBinSize_.reserve(4);
	}
	//  &&& Should these sizes be computed on the fly from other 
	//  &&& variables (which are currently not stored in this object,
	//  &&& but maybe should be?)

	inline void push_back( DbEntry e) {
		errors_.push_back(e);
	}
	
	inline void push_back_bin( DbEntryBinSize e) {
		errorsBinSize_.push_back(e);
	}

	inline void set_version (float v) {
		version = v;
	}

	// &&& Should we be able to read this from an iostream?  See PxCPEdbUploader...

 private:
	DbVector errors_ ;
	DbBinSizeVector errorsBinSize_;
	float version;
};

#endif
