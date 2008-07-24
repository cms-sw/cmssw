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


class SiStripSummary {

	public:

		struct DetRegistry{
			uint32_t detid;
			uint32_t ibegin;
		};
                
		enum TrackerRegion { TRACKER = 0, TIB = 1, TOB = 2, TID = 3, TEC = 4, 
		                     TIB_1 = 10, TIB_2 = 11, TIB_3 = 12, TIB_4 = 13,
		                     TOB_1 = 14, TOB_2 = 15, TOB_3 = 16, TOB_4 = 17, TOB_5 = 18, TOB_6 =19, 
				     TID_1 = 20, TID_2 = 21, TID_3 = 22,
				     TEC_1 = 23, TEC_2 = 24, TEC_3 = 25, TEC_4 = 26, TEC_5 = 27, TEC_6 = 28, TEC_7 = 29, TEC_8 = 30, TEC_9 = 31
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
                
		
		SiStripSummary(std::vector<std::string>& userDBContent, std::string tag);
		SiStripSummary(const SiStripSummary& input);
		SiStripSummary(){};
		~SiStripSummary(){};
		

                // RETURNS POSITION OF DETID IN v_sum_
		//
		const Range getRange(const uint32_t& detID) const;
		
		
		// RETURNS LIST OF DETIDS 
		//
		std::vector<uint32_t> getDetIds() const;
                
		
		// INSERT SUMMARY OBJECTS...
		//
		bool put(const uint32_t& detID, InputVector &input, std::vector<std::string>& userContent );
		bool put(TrackerRegion region, InputVector &input, std::vector<std::string>& userContent );
		void setObj(const uint32_t& detID, std::string elementName, float value);
		
		
		// RETRIEVE SUMMARY OBJECTS...
		//
		// returns info "elementName" for a given detId
		float getSummaryObj(uint32_t& detID, std::string elementName) const;	
		
		// returns a vector of selected infos related to a given detId 
		std::vector<float> getSummaryObj(uint32_t& detID, std::vector<std::string> list) const; 
		 
		// returns a vector filled with "info elementName" for each detId 
		// The order is SORTED according to the one used in getDetIds() !
		std::vector<float> getSummaryObj(std::string elementName) const;     
		                                                      
		// returns the entire SummaryObj related to one detId
		std::vector<float> getSummaryObj(uint32_t& detID) const;
			
		// returns everything, all SummaryObjects for all detIds (unsorted !)
		std::vector<float> getSummaryObj() const;		      
		
	
		// INLINE METHODS ABOUT RUN, TIME VALUE...
		//
                inline void setUserDBContent(std::vector<std::string> userDBContent)  { userDBContent_ = userDBContent;}
	        inline void setRunNr(int inputRunNr)                       { runNr_ = inputRunNr;      }
                inline void setTimeValue(unsigned long long inputTimeValue){ timeValue_=inputTimeValue;}
                inline void setTag(std::string tag)                        { tag_ = tag;               }
                
		inline unsigned long long getTimeValue() const             { return timeValue_;        }
                inline std::vector<std::string>  getUserDBContent() const  { return userDBContent_;    }
                inline std::string  getTag() const                         { return tag_;              }
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
		std::string tag_;
		
		
        protected:	
	
	        // RETURNS POSITION OF ELEMENTNAME IN userDBContent_
	        const size_t getPosition(std::string elementName) const;
	
	
   };
		

#endif
