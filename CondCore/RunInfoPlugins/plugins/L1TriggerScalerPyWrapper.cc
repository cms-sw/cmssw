
#include "CondFormats/RunInfo/interface/L1TriggerScaler.h"

#include "CondCore/Utilities/interface/PayLoadInspector.h"
#include "CondCore/Utilities/interface/InspectorPythonWrapper.h"

#include <string>
#include <fstream>

namespace cond {

  template<>
  class ValueExtractor<L1TriggerScaler>: public  BaseValueExtractor<L1TriggerScaler> {
  public:

    typedef L1TriggerScaler Class;
    typedef ExtractWhat<Class> What;
    static What what() { return What();}

    ValueExtractor(){}
    ValueExtractor(What const & what)
    {
      // here one can make stuff really complicated...
    }
    void compute(Class const & it){
    }
  private:
  
  };


  template<>
  std::string
  PayLoadInspector<L1TriggerScaler>::dump() const {
    std::stringstream ss;
    return ss.str();
    
  }
  
  template<>
  std::string PayLoadInspector<L1TriggerScaler>::summary() const {
    std::stringstream ss;
    ss << object().m_run.size() <<", ";
    if (!object().m_run.empty()) {
      ss << object().m_run.front().m_rn;
      ss << ", ";
      ss << object().m_run.front().m_lumisegment;
      ss << ", " << object().m_run.front().m_date;
 ss << ", " << object().m_run.front().m_date;

      for(size_t i=0; i<object().m_run.front().m_GTAlgoCounts.size(); i++ ){ 
      ss << "m_GTAlgoCounts["<<i<<"] = "<< object().m_run.front().m_GTAlgoCounts[i]<<std::endl;  
    }
      for(size_t i=0; i<object().m_run.front().m_GTAlgoRates.size(); i++ ){ 
      ss << "m_GTAlgoRates["<<i<<"] = "<< object().m_run.front().m_GTAlgoRates[i]<<std::endl;  
    }
    for(size_t i=0; i<object().m_run.front().m_GTAlgoPrescaling.size(); i++ ){ 
      ss << "m_GTAlgoPrescaling["<<i<<"] = "<< object().m_run.front().m_GTAlgoPrescaling[i]<<std::endl;  
    } 
    for(size_t i=0; i<object().m_run.front().m_GTTechCounts.size(); i++ ){ 
      ss << " m_GTTechCounts["<<i<<"] = "<< object().m_run.front().m_GTTechCounts[i]<<std::endl;  
    } 
    for(size_t i=0; i<object().m_run.front().m_GTTechRates.size(); i++ ){ 
      ss << " m_GTTechRates["<<i<<"] = "<< object().m_run.front().m_GTTechRates[i]<<std::endl;  
    } 
    for(size_t i=0; i<object().m_run.front().m_GTTechPrescaling.size(); i++ ){ 
      ss << " m_GTTechPrescaling["<<i<<"] = "<< object().m_run.front().m_GTTechPrescaling[i]<<std::endl;  
    } 
    for(size_t i=0; i<object().m_run.front().m_GTPartition0TriggerCounts.size(); i++ ){ 
      ss << " m_GTPartition0TriggerCounts["<<i<<"] = "<< object().m_run.front().m_GTPartition0TriggerCounts[i]<<std::endl;  
    } 
    for(size_t i=0; i<object().m_run.front().m_GTPartition0TriggerRates.size(); i++ ){ 
      ss << " m_GTPartition0TriggerRates["<<i<<"] = "<< object().m_run.front().m_GTPartition0TriggerRates[i]<<std::endl;  
    } 
    for(size_t i=0; i<object().m_run.front().m_GTPartition0DeadTime.size(); i++ ){ 
      ss << " m_GTPartition0DeadTime["<<i<<"] = "<< object().m_run.front().m_GTPartition0DeadTime[i]<<std::endl;  
    }
    for(size_t i=0; i<object().m_run.front().m_GTPartition0DeadTimeRatio.size(); i++ ){ 
      ss << " m_GTPartition0DeadTimeRatio["<<i<<"] = "<< object().m_run.front().m_GTPartition0DeadTimeRatio[i]<<std::endl;  
    } 
  
 
      ss << "; ";
      ss << object().m_run.back().m_rn;
      ss << ", ";
      ss << object().m_run.back().m_lumisegment;
      ss << ", " << object().m_run.back().m_date;

     for(size_t i=0; i<object().m_run.back().m_GTAlgoCounts.size(); i++ ){ 
      ss << "m_GTAlgoCounts["<<i<<"] = "<< object().m_run.back().m_GTAlgoCounts[i]<<std::endl;  
    }
      for(size_t i=0; i<object().m_run.back().m_GTAlgoRates.size(); i++ ){ 
      ss << "m_GTAlgoRates["<<i<<"] = "<< object().m_run.back().m_GTAlgoRates[i]<<std::endl;  
    }
    for(size_t i=0; i<object().m_run.back().m_GTAlgoPrescaling.size(); i++ ){ 
      ss << "m_GTAlgoPrescaling["<<i<<"] = "<< object().m_run.back().m_GTAlgoPrescaling[i]<<std::endl;  
    } 
    for(size_t i=0; i<object().m_run.back().m_GTTechCounts.size(); i++ ){ 
      ss << " m_GTTechCounts["<<i<<"] = "<< object().m_run.back().m_GTTechCounts[i]<<std::endl;  
    } 
    for(size_t i=0; i<object().m_run.back().m_GTTechRates.size(); i++ ){ 
      ss << " m_GTTechRates["<<i<<"] = "<< object().m_run.back().m_GTTechRates[i]<<std::endl;  
    } 
    for(size_t i=0; i<object().m_run.back().m_GTTechPrescaling.size(); i++ ){ 
      ss << " m_GTTechPrescaling["<<i<<"] = "<< object().m_run.back().m_GTTechPrescaling[i]<<std::endl;  
    } 
    for(size_t i=0; i<object().m_run.back().m_GTPartition0TriggerCounts.size(); i++ ){ 
      ss << " m_GTPartition0TriggerCounts["<<i<<"] = "<< object().m_run.back().m_GTPartition0TriggerCounts[i]<<std::endl;  
    } 
    for(size_t i=0; i<object().m_run.back().m_GTPartition0TriggerRates.size(); i++ ){ 
      ss << " m_GTPartition0TriggerRates["<<i<<"] = "<< object().m_run.back().m_GTPartition0TriggerRates[i]<<std::endl;  
    } 
    for(size_t i=0; i<object().m_run.back().m_GTPartition0DeadTime.size(); i++ ){ 
      ss << " m_GTPartition0DeadTime["<<i<<"] = "<< object().m_run.back().m_GTPartition0DeadTime[i]<<std::endl;  
    }
    for(size_t i=0; i<object().m_run.back().m_GTPartition0DeadTimeRatio.size(); i++ ){ 
      ss << " m_GTPartition0DeadTimeRatio["<<i<<"] = "<< object().m_run.back().m_GTPartition0DeadTimeRatio[i]<<std::endl;  
    } 
    }
    return ss.str();
  }
  

  template<>
  std::string PayLoadInspector<L1TriggerScaler>::plot(std::string const & filename,
						   std::string const &, 
						   std::vector<int> const&, 
						   std::vector<float> const& ) const {
    std::string fname = filename + ".png";
    std::ofstream f(fname.c_str());
    return fname;
  }


}

PYTHON_WRAPPER(L1TriggerScaler,L1TriggerScaler);
