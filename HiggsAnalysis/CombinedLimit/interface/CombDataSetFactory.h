#ifndef HiggsAnalysis_CombinedLimit_CombDataSetFactory_h
#define HiggsAnalysis_CombinedLimit_CombDataSetFactory_h

#include <RooCategory.h>
#include <RooArgSet.h>
#include <RooDataHist.h>
#include <RooDataSet.h>
#include <RooRealVar.h>
#include <map>
#include <string>


//_________________________________________________
/*
BEGIN_HTML
CombDataSetFactory is helper class for creating combined datasets since PyROOT can't call some constructors of RooDataHist 
END_HTML
*/
//
class CombDataSetFactory : public TObject {

   public:
      CombDataSetFactory() {}
      CombDataSetFactory(const RooArgSet &vars, RooCategory &cat) ;
      ~CombDataSetFactory() ;

      void addSetBin(const char *label, RooDataHist *set);
      void addSetAny(const char *label, RooDataSet  *set);
      void addSetAny(const char *label, RooDataHist *set);

      RooDataHist *done(const char *name, const char *title) ;
      RooDataSet *doneUnbinned(const char *name, const char *title) ;

      ClassDef(CombDataSetFactory,1) // Make RooDataHist

    private:
        RooArgSet vars_;
        RooCategory *cat_;
        RooRealVar  *weight_;
        std::map<std::string, RooDataHist *> map_;
        std::map<std::string, RooDataSet *> mapUB_;
};

#endif
