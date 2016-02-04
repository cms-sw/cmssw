#ifndef L1TriggerScaler_h
#define L1TriggerScaler_h

#include <iostream>
#include<vector>

/*
 *  \class L1TriggerScaler
 *  
 *  hosting L1TriggerScaler information
 *
 *  \author Michele de Gruttola (degrutto) - INFN Naples / CERN (August-21-2008)
 *
*/


class L1TriggerScaler {
public:
  // a struct with the information for each lumi

  struct Lumi{
    Lumi(){}
    ~Lumi(){}
    int m_runnumber;
    long long m_lumi_id;
    std::string m_start_time; 
    // std::string m_date_str;
    //std::string  m_string_value;
    std::string  m_string_format;
    long m_rn;
    int m_lumisegment;
    // std::string m_version;
    //std::string m_context;
    std::string m_date; 
   
    
   
    
    std::vector<int> m_GTAlgoCounts;   
    // m_GTAlgoCounts.reserve(128);
    
    std::vector<float> m_GTAlgoRates;
    
    std::vector<int> m_GTAlgoPrescaling;
   
    std::vector<int> m_GTTechCounts;
   
    std::vector<float> m_GTTechRates;
   
    std::vector<int> m_GTTechPrescaling;
   
    std::vector<int> m_GTPartition0TriggerCounts;
   
    std::vector<float> m_GTPartition0TriggerRates;
    
    std::vector<int> m_GTPartition0DeadTime;
    
    std::vector<float> m_GTPartition0DeadTimeRatio;
    
   };
   
// the fondamental object is a vector of struct Lumi   
  L1TriggerScaler();
  virtual ~L1TriggerScaler(){}
  // fondamental object   
  std::vector<Lumi> m_run;
    
 
   // printing everything
  void printAllValues() const;
  void printRunValue()const;
  void printLumiSegmentValues() const;
  void printFormat() const;
  void printGTAlgoCounts() const;
  void printGTAlgoRates() const;
  void printGTAlgoPrescaling() const;
  void printGTTechCounts() const;
  void printGTTechRates() const;
  void printGTTechPrescaling() const;
  void printGTPartition0TriggerCounts() const;
  void printGTPartition0TriggerRates() const;
  void printGTPartition0DeadTime() const;
  void printGTPartition0DeadTimeRatio() const;  

  typedef std::vector<Lumi>::const_iterator LumiIterator;
 


 
 


 

void SetRunNumber(int n) {
    m_runnumber= n;
  }


 private:
 int  m_lumisegment; 
 int m_runnumber;
 // std::string  m_string_value;
 //std::vector<int>  m_GTAlgoCounts;
};

#endif
