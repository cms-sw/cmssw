#ifndef CondFormats_SiPixelCPEParmErrors_h
#define CondFormats_SiPixelCPEParmErrors_h 1

#include <vector>

//--- Maybe should make this a class const, but that'd be too much work.
//--- This usage is not worth it, since in the debugger will be obvious
//--- what this is! ;)
#define NONSENSE -99999.9

class SiPixelCPEParmErrors {
 public:
        //! A struct to hold information for a given (alpha,beta,size)
        struct DbEntry {
		float sigma;
		float rms;
		float bias;
		float pix_height;
		float ave_qclu;
	  DbEntry() : sigma(NONSENSE), rms(NONSENSE), 
		    bias(NONSENSE), pix_height(NONSENSE), 
		    ave_qclu(NONSENSE) {}
	  ~DbEntry() {}
	};
	typedef std::vector<DbEntry> DbVector;

	SiPixelCPEParmErrors() : errorsBx_(), errorsBy_(), errorsFx_(), errorsFy_() {}
	virtual ~SiPixelCPEParmErrors(){}

	//!  Accessors for the vectors -- non-const version
	inline DbVector & errorsBx() { return errorsBx_ ; }
	inline DbVector & errorsBy() { return errorsBy_ ; }
	inline DbVector & errorsFx() { return errorsFx_ ; }
	inline DbVector & errorsFy() { return errorsFy_ ; }

	//!  Accessors for the vectors -- const version
	inline const DbVector & errorsBx() const { return errorsBx_ ; }
	inline const DbVector & errorsBy() const { return errorsBy_ ; }
	inline const DbVector & errorsFx() const { return errorsFx_ ; }
	inline const DbVector & errorsFy() const { return errorsFy_ ; }

	//!  Reserve some reasonable sizes for the vectors. 
	inline void reserve() {
	  errorsBx_.reserve(300);
	  errorsBy_.reserve(300);
	  errorsFx_.reserve(300);
	  errorsFy_.reserve(300);
	}
	//  &&& Should these sizes be computed on the fly from other 
	//  &&& variables (which are currently not stored in this object,
	//  &&& but maybe should be?)


	//!  Store a new DbEntry, depending on the detector type.
	inline void push_back( int det_type, DbEntry e) {
	  switch (det_type) {
	  case 1: errorsBx_.push_back(e); break;
	  case 2: errorsBy_.push_back(e); break;
	  case 3: errorsFx_.push_back(e); break;
	  case 4: errorsFy_.push_back(e); break;
	  default: // throw something?
	    assert(det_type > 0 && det_type < 5 );
	  }
	}


	// &&& Should we be able to read this from an iostream?  See PxCPEdbUploader...

 private:
	DbVector errorsBx_ ;
	DbVector errorsBy_ ;
	DbVector errorsFx_ ;
	DbVector errorsFy_ ;
};

#endif
