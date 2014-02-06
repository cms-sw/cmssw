#ifndef DPGAnalysis_SiStripTools_ClusMultPlots_h
#define DPGAnalysis_SiStripTools_ClusMultPlots_h

class TH1D;
class TFile;

void ClusMultPlots(const char* fullname, const char* pxmod, const char* strpmod, const char* corrmod,
		   const char* pxlabel, const char* strplabel, const char* corrlabel, const char* postfix, const char* shortname, const char* outtrunk);
void ClusMultInvestPlots(const char* fullname, const char* mod, const char* label, const char* postfix, const char* subdet, const char* shortname, const char* outtrunk);
void ClusMultCorrPlots(const char* fullname, const char* mod, const char* label, const char* postfix, const char* shortname, const char* outtrunk);
void ClusMultVtxCorrPlots(const char* fullname, const char* mod, const char* label, const char* postfix, const char* subdet, const char* shortname, const char* outtrunk);
void ClusMultLumiCorrPlots(const char* fullname, const char* mod, const char* label,const char* postfix, const char* subdet, const char* shortname, const char* outtrunk);

#endif  //  DPGAnalysis_SiStripTools_ClusMultPlots_h
