#ifndef HiggsAnalysis_CombinedLimit_CombDataSetFactory_h
#define HiggsAnalysis_CombinedLimit_CombDataSetFactory_h

#include <RooCategory.h>
#include <RooArgSet.h>
#include <RooDataHist.h>
#include <RooDataSet.h>
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

      void addSet(const char *label, RooDataHist *set);
      void addSet(const char *label, RooDataSet *set);

      RooDataHist *done(const char *name, const char *title) ;
      RooDataSet *doneUnbinned(const char *name, const char *title) ;

      ClassDef(CombDataSetFactory,1) // Make RooDataHist

    private:
        RooArgSet vars_;
        RooCategory *cat_;
        std::map<std::string, RooDataHist *> map_;
        std::map<std::string, RooDataSet *> mapUB_;
};

#endif
