#ifndef SiStripSummary_h
#define SiStripSummary_h

#include<vector>
#include<map>
#include<iostream>
#include<boost/cstdint.hpp>
#include "FWCore/Utilities/interface/Exception.h"


class SiStripSummary {

	public:

		struct DetRegistry{
			uint32_t detid;
			uint32_t ibegin;
			uint32_t iend;
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


		typedef std::vector<float>::const_iterator               ContainerIterator;  
		typedef std::pair<ContainerIterator, ContainerIterator>  Range; 
		     
		typedef std::vector<DetRegistry>                         Registry;
		typedef Registry::const_iterator                         RegistryIterator;
		
		typedef std::vector<float>		                 InputVector;
                
		
		SiStripSummary(std::vector<std::string>& userDBContent, std::string tag);
		SiStripSummary(const SiStripSummary& input);
		SiStripSummary(){};
		~SiStripSummary(){};

		
		const Range getRange(const uint32_t& detID) const;
		std::vector<uint32_t> getDetIds() const;
                
		ContainerIterator getDataVectorBegin()     const {return v_sum.begin();  }
		ContainerIterator getDataVectorEnd()       const {return v_sum.end();    } 
		RegistryIterator  getRegistryVectorBegin() const {return indexes.begin();}
		RegistryIterator  getRegistryVectorEnd()   const {return indexes.end();  }
                
		
		// insert summary objects
		//
		bool put(const uint32_t& detID, InputVector &input, std::vector<std::string>& userContent );
		bool put(const uint32_t& detID, InputVector &input);
		bool put(const uint32_t& detID, float input);
		bool put(TrackerRegion region, InputVector &input);
		bool put(TrackerRegion region, InputVector &input, std::vector<std::string>& userContent );
		void setSummaryObj(const uint32_t& detID, std::vector<float>& SummaryObj);	       
		void setObj(const uint32_t& detID, std::string elementName, float value);
		
		
		// retrieve summary objects
		//
		// returns one info for one detId
		float getSummaryObj(uint32_t& detID, std::string elementName) const;	
		
		// returns a vector of selected infos related to one detId 
		std::vector<float> getSummaryObj(uint32_t& detID, std::vector<std::string> list) const; 
		 
		// returns a vector filled with "info elementName" for each detId 
		// The order is SORTED according to the one used in getDetIds() !
		std::vector<float> getSummaryObj(std::string elementName) const;     
		                                                      
		// returns the entire SummaryObj related to one detId
		std::vector<float> getSummaryObj(uint32_t& detID) const;
			
		// returns everything, all SummaryObjects for all detIds (unsorted !)
		std::vector<float> getSummaryObj() const;		      
		
	
		void setData(float summaryInfo, std::vector<float>& v); // to be kept ?
                
                inline void  setUserDBContent(std::vector<std::string> userDBContent)  { userDBContent_=userDBContent;    }

		// inline methods about run, time value ...
		//
	        inline void setRunNr(int inputRunNr)              { runNr_ = inputRunNr;      }
                inline void setTimeValue(unsigned long long inputTimeValue){ timeValue_=inputTimeValue;}
                inline void setTag(std::string tag)                        { tag_ = tag;               }
                
		inline unsigned long long getTimeValue() const             { return timeValue_;        }
                inline std::vector<std::string>  getUserDBContent() const  { return userDBContent_;    }
                inline std::string  getTag() const                         { return tag_;              }
                inline int getRunNr() const                       { return runNr_;            }
               
                void print();

		std::vector<std::string>        userDBContent_;
                std::vector<float> 	        v_sum; 
		std::vector<DetRegistry> 	indexes;
		
	        int runNr_;
                unsigned long long timeValue_;
		std::string tag_;
		
		
        protected:	
	
	        const size_t getPosition(std::string elementName) const;
	
	
   };
		

#endif
