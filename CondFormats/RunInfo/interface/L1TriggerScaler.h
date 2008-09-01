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
  // a struct with the information for each lumisection

  struct Lumi{
    Lumi(){}
    ~Lumi(){}
    int m_runnumber;
    long long m_lumi_id;
    std::string  m_string_format;
    long m_rn;
    int m_lumisegment;
    std::string m_date; 
    
    std::vector<int> m_GTAlgoCounts;   
    
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
 
 private:
 int  m_lumisegment; 
 int m_runnumber;

};

#endif
