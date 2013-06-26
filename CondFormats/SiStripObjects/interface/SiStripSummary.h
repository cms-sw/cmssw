#ifndef SiStripSummary_h
#define SiStripSummary_h

#include<vector>
#include<map>
#include<iostream>
#include<boost/cstdint.hpp>
#include "FWCore/Utilities/interface/Exception.h"

/**
 @class SiStripSummary
 @author D. Giordano, A.-C. Le Bihan
 @class to hold historic DQM summary informations
*/
 
namespace sistripsummary {
  enum TrackerRegion { TRACKER = 0, 
		       TIB = 1, 
		       TIB_1 = 11, TIB_2 = 12, TIB_3 = 13, TIB_4 = 14,
		       TOB = 2, 
		       TOB_1 = 21, TOB_2 = 22, TOB_3 = 23, TOB_4 = 24, TOB_5 = 25, TOB_6 = 26, 
		       TID = 3, 
		       TIDM = 31, 
		       TIDP = 32, 
		       TIDM_1 = 311, TIDM_2 = 312, TIDM_3 = 313,
		       TIDP_1 = 321, TIDP_2 = 322, TIDP_3 = 323,
		       TEC = 4, 
		       TECM = 41, 
		       TECP = 42, 
		       TECM_1 = 411, TECM_2 = 412, TECM_3 = 413, TECM_4 = 414, TECM_5 = 415, TECM_6 = 416, TECM_7 = 417, TECM_8 = 418, TECM_9 = 419,
		       TECP_1 = 421, TECP_2 = 422, TECP_3 = 423, TECP_4 = 424, TECP_5 = 425, TECP_6 = 426, TECP_7 = 427, TECP_8 = 428, TECP_9 = 429
  };
}

class SiStripSummary {

	public:

		struct DetRegistry{
			uint32_t detid;
			uint32_t ibegin;
		};
                
		class StrictWeakOrdering{
			public:
				bool operator() (const DetRegistry& p,const uint32_t& i) const {return p.detid < i;}
		};


                // SOME DEFINITIONS
                //
	        typedef std::vector<float>::const_iterator               ContainerIterator;  
		typedef std::pair<ContainerIterator, ContainerIterator>  Range; 		     
		typedef std::vector<DetRegistry>                         Registry;
		typedef Registry::const_iterator                         RegistryIterator;
		typedef std::vector<float>		                 InputVector;
                
		
		SiStripSummary(std::vector<std::string>& userDBContent);
		SiStripSummary(const SiStripSummary& input);
		SiStripSummary(){};
		~SiStripSummary(){};
		
             
		ContainerIterator getDataVectorBegin()     const {return v_sum_.begin();  }
		ContainerIterator getDataVectorEnd()       const {return v_sum_.end();    } 
		RegistryIterator  getRegistryVectorBegin() const {return indexes_.begin();}
		RegistryIterator  getRegistryVectorEnd()   const {return indexes_.end();  }

                // RETURNS POSITION OF DETID IN v_sum_
		//
		const Range getRange(const uint32_t& detID) const;
		
		
		// RETURNS LIST OF DETIDS 
		//
		std::vector<uint32_t> getDetIds() const;
                
		
		// INSERT SUMMARY OBJECTS...
		//
		bool put(const uint32_t& detID, InputVector &input, std::vector<std::string>& userContent );
		bool put(sistripsummary::TrackerRegion region, InputVector &input, std::vector<std::string>& userContent );
		void setObj(const uint32_t& detID, std::string elementName, float value);
		
		
		// RETRIEVE SUMMARY OBJECTS...
		//
		
		// returns a vector of selected infos related to a given detId 
		std::vector<float> getSummaryObj(uint32_t& detID, const std::vector<std::string>& list) const; 
		std::vector<float> getSummaryObj(sistripsummary::TrackerRegion region,const std::vector<std::string>& list) const; 
		 
		// returns a vector filled with "info elementName" for each detId 
		// The order is SORTED according to the one used in getDetIds() !
		std::vector<float> getSummaryObj(std::string elementName) const;     
		                                                      
		// returns the entire SummaryObj related to one detId
		std::vector<float> getSummaryObj(uint32_t& detID) const;
			
		// returns everything, all SummaryObjects for all detIds (unsorted !)
		std::vector<float> getSummaryObj() const;		      
		
	
		// INLINE METHODS ABOUT RUN, TIME VALUE...
		//
                inline void setUserDBContent(const std::vector<std::string>& userDBContent)  { userDBContent_ = userDBContent;}
	        inline void setRunNr(int inputRunNr)                       { runNr_ = inputRunNr;      }
                inline void setTimeValue(unsigned long long inputTimeValue){ timeValue_=inputTimeValue;}
                
		inline unsigned long long getTimeValue() const             { return timeValue_;        }
                inline std::vector<std::string>  getUserDBContent() const  { return userDBContent_;    }
                inline int getRunNr() const                                { return runNr_;            }
               
	       
	        // PRINT METHOD...
		//
                void print();
                
		
		// SISTRIPSUMMARY MEMBERS...
		//
		std::vector<std::string>        userDBContent_;
                std::vector<float> 	        v_sum_; 
		std::vector<DetRegistry> 	indexes_;
		
	        int runNr_; 
                unsigned long long timeValue_;
		
		
        protected:	
	
	        // RETURNS POSITION OF ELEMENTNAME IN userDBContent_
	        const short getPosition(std::string elementName) const;
	
	
   };
		

#endif
