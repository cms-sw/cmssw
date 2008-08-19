#ifndef __LUMIFILEWRITER_H__
#define __LUMIFILEWRITER_H__


class LumiFileWriter : public edm::EDAnalyzer {
   public:
      explicit LumiFileWriter(const edm::ParameterSet&);
      ~LumiFileWriter();


   private:
      virtual void beginJob(const edm::EventSetup&) ;
      virtual void analyze(const edm::Event&, const edm::EventSetup&);
      virtual void endJob() ;

 
      HCAL_HLX::TCPReceiver      HLXTCP;  
      HCAL_HLX::LUMI_SECTION     localSection;
      HCAL_HLX::ROOTSchema       lumiSchema;
      HCAL_HLX::ROOTFileMerger   RFM;
      HCAL_HLX::ROOTFileTransfer RFT;
      HCAL_HLX::HTMLGenerator    webPage;
      
      unsigned int reconTime;

      bool bMerge_;
      bool bWBM_;
      bool bTransfer_;
      bool bTest_;

      unsigned int lastRun_;
      bool lastCMSLive_;

      unsigned int LSCount_;
      unsigned int MergeRate_;
      
};

#endif
