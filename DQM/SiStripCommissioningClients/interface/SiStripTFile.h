// Last commit: $Id: SiStripTFile.h,v 1.2 2008/02/21 14:19:03 bainbrid Exp $

#ifndef DQM_SiStripCommissioningClients_SiStripTFile_H
#define DQM_SiStripCommissioningClients_SiStripTFile_H

#include "DataFormats/SiStripCommon/interface/SiStripConstants.h" 
#include "TFile.h"
#include <vector>
#include <string>
#include <map>

class TDirectory;
class TH1;

/** 
    @class : SiStripTFile 
    @author : M.Wingham 

    @brief : Adds functionality to TFile to ease building and
    navigation of TFiles containing DQM histograms.
*/
class SiStripTFile : public TFile {

 public:

  // -------------------- Constructors, destructors, typedefs --------------------
  
  /** Constructor */
  SiStripTFile( const char* fname, 
		Option_t* option = "UPDATE", 
		const char* ftitle = "", 
		Int_t compress = 1 );
  
  /** Destructor */
  virtual ~SiStripTFile();
  
  /** */
  typedef std::vector<TH1*> Histos;

  /** */
  typedef std::map< std::string, Histos > HistosMap;
  
  // -------------------- Public interface --------------------
  
  /** Formats the commissioning file with the correct "top-level"
      directory structure. Inserts string defining commissioning RunType
      in sistrip::root_ directory */
  TDirectory* setDQMFormat( sistrip::RunType, sistrip::View );
  
  /** Checks file complies with DQM format requirements. If so,
      updates record directory "top-level" directory structure and of
      readout view and commissioning RunType. */
  TDirectory* readDQMFormat();
  
  /** Checks to see if the file complies with DQM format
      requirements. */
  bool queryDQMFormat();
  
  /** Returns the "top" directory (describing the readout view) */
  TDirectory* top();
  
  /** Returns the dqm directory */
  TDirectory* dqmTop();
  
  /** Returns the sistrip::root_ directory */
  TDirectory* sistripTop();

  /** Get Method */
  sistrip::RunType& runType();

  /** Get Method */
  sistrip::View& View();

  /** Adds the directory paths for the device of given key. Must use dqmFormat() before this method. */
  void addDevice(unsigned int key);

  /** Adds a path to the file. Any directories within the path that already exist are not recreated.*/
  TDirectory* addPath( const std::string& );

  /** Finds TH1 histograms, iterating through sub-directories. Fills a map, indexed by the histogram path. */
  void findHistos(TDirectory*, std::map< std::string, std::vector<TH1*> >*);

  /** Finds histos and sub-dirs found within given directory. Updates
      map with found histos, indexed by dir path. */
  void dirContent(TDirectory*, std::vector<TDirectory*>*, std::map< std::string, std::vector<TH1*> >*);

 private:

  /** RunType */
  sistrip::RunType runType_;

  /** Logical view. */
  sistrip::View view_;

  /** Readout view directory */
  TDirectory* top_;

  /** dqm directory */
  TDirectory* dqmTop_;

  /** sistrip::root_ directory */
  TDirectory* sistripTop_;

  /** True if dqmFormat() operation has been performed */
  bool dqmFormat_;

};

#endif // DQM_SiStripCommissioningClients_SiStripTFile_H
