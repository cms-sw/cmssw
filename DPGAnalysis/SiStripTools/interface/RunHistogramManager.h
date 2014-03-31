#ifndef DPGAnalysis_SiStripTools_RunHistogramManager_H
#define DPGAnalysis_SiStripTools_RunHistogramManager_H

#include <vector>
#include <map>
#include <string>
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Framework/interface/Run.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "CommonTools/UtilAlgos/interface/TFileService.h"
#include "DataFormats/Common/interface/ConditionsInEdm.h"
#include "TH2F.h"
#include "TProfile2D.h"

class TH1F;
class TProfile;

 class BaseHistoParams{

 public:
   BaseHistoParams();
   virtual ~BaseHistoParams();

   //   virtual void beginRun(const edm::Run& iRun, TFileDirectory& subrun);
   virtual void beginRun(const unsigned int irun, TFileDirectory& subrun, const char* fillrun) = 0;

 };

template <class T>
  class HistoParams: public BaseHistoParams
{

 public:
  HistoParams(T** pointer, const std::string type, const std::string name, const std::string title,
	      const unsigned int nbinx=-1, const double xmin = -1., const double xmax = -1.,
	      const unsigned int nbiny=-1, const double ymin = -1., const double ymax = -1.):
    BaseHistoParams(),
    _pointer(pointer),
    _type(type), _name(name), _title(title), _nbinx(nbinx), _xmin(xmin), _xmax(xmax),
    _nbiny(nbiny), _ymin(ymin), _ymax(ymax), _runpointers() { }

    ~HistoParams() {

      delete _pointer;
      LogDebug("Destructor") << "Destroy " << _name;

    }

    virtual void beginRun(const unsigned int irun, TFileDirectory& subrun, const char* fillrun) {

      if(_runpointers.find(irun)!=_runpointers.end()) {
	*_pointer = _runpointers[irun];
	LogDebug("TH1Fbooked") << "Histogram " << _name.c_str() << " already exists " << _runpointers[irun];

      }
      else {

	char title[400];
	sprintf(title,"%s %s %d",_title.c_str(),fillrun,irun);

	_runpointers[irun] = subrun.make<T>(_name.c_str(),
						  title,
						  _nbinx,
						  _xmin,
						  _xmax);

	*_pointer = _runpointers[irun];
	LogDebug("TH1Fbooked") << "Histogram " << _name.c_str() << " booked " << _runpointers[irun];
      }

    }

 private:
    T** _pointer;
    std::string _type;
    std::string _name;
    std::string _title;
    unsigned int _nbinx;
    double _xmin;
    double _xmax;
    unsigned int _nbiny;
    double _ymin;
    double _ymax;
    std::map<unsigned int, T*> _runpointers;

 };

template <>
  class HistoParams<TH2F>: public BaseHistoParams
{

 public:
  HistoParams(TH2F** pointer, const std::string type, const std::string name, const std::string title,
	      const unsigned int nbinx=-1, const double xmin = -1., const double xmax = -1.,
	      const unsigned int nbiny=-1, const double ymin = -1., const double ymax = -1.):
    BaseHistoParams(),
    _pointer(pointer),
    _type(type), _name(name), _title(title), _nbinx(nbinx), _xmin(xmin), _xmax(xmax),
    _nbiny(nbiny), _ymin(ymin), _ymax(ymax), _runpointers() { }


    ~HistoParams() {

      delete _pointer;
      LogDebug("TH2FDestructor") << "Destroy " << _name;

    }

    virtual void beginRun(const unsigned int irun, TFileDirectory& subrun, const char* fillrun) {

      if(_runpointers.find(irun)!=_runpointers.end()) {
	*_pointer = _runpointers[irun];
	LogDebug("TH2Fbooked") << "Histogram " << _name.c_str() << " already exists " << _runpointers[irun];

      }
      else {

	char title[400];
	sprintf(title,"%s %s %d",_title.c_str(),fillrun,irun);

	_runpointers[irun] = subrun.make<TH2F>(_name.c_str(),
						  title,
						  _nbinx,
						  _xmin,
						  _xmax,
						  _nbiny,
						  _ymin,
						  _ymax);

	*_pointer = _runpointers[irun];
	LogDebug("TH2Fbooked") << "Histogram " << _name.c_str() << " booked " << _runpointers[irun];
      }


    }

 private:
    TH2F** _pointer;
    std::string _type;
    std::string _name;
    std::string _title;
    unsigned int _nbinx;
    double _xmin;
    double _xmax;
    unsigned int _nbiny;
    double _ymin;
    double _ymax;
    std::map<unsigned int, TH2F*> _runpointers;

 };

template <>
  class HistoParams<TProfile2D>: public BaseHistoParams
{

 public:
  HistoParams(TProfile2D** pointer, const std::string type, const std::string name, const std::string title,
	      const unsigned int nbinx=-1, const double xmin = -1., const double xmax = -1.,
	      const unsigned int nbiny=-1, const double ymin = -1., const double ymax = -1.):
    BaseHistoParams(),
    _pointer(pointer),
    _type(type), _name(name), _title(title), _nbinx(nbinx), _xmin(xmin), _xmax(xmax),
    _nbiny(nbiny), _ymin(ymin), _ymax(ymax), _runpointers() { }


    ~HistoParams() {

      delete _pointer;
      LogDebug("TProfile2DDestructor") << "Destroy " << _name;

    }

    virtual void beginRun(const unsigned int irun, TFileDirectory& subrun, const char* fillrun) {

      if(_runpointers.find(irun)!=_runpointers.end()) {
	*_pointer = _runpointers[irun];
	LogDebug("TProfile2Dbooked") << "Histogram " << _name.c_str() << " already exists " << _runpointers[irun];

      }
      else {

	char title[400];
	sprintf(title,"%s %s %d",_title.c_str(),fillrun,irun);

	_runpointers[irun] = subrun.make<TProfile2D>(_name.c_str(),
						  title,
						  _nbinx,
						  _xmin,
						  _xmax,
						  _nbiny,
						  _ymin,
						  _ymax);

	*_pointer = _runpointers[irun];
	LogDebug("TProfile2Dbooked") << "Histogram " << _name.c_str() << " booked " << _runpointers[irun];
      }


    }

 private:
    TProfile2D** _pointer;
    std::string _type;
    std::string _name;
    std::string _title;
    unsigned int _nbinx;
    double _xmin;
    double _xmax;
    unsigned int _nbiny;
    double _ymin;
    double _ymax;
    std::map<unsigned int, TProfile2D*> _runpointers;

 };


class RunHistogramManager {

 public:

  RunHistogramManager(edm::ConsumesCollector&& iC, const bool fillHistograms=false);
  RunHistogramManager(edm::ConsumesCollector& iC, const bool fillHistograms=false);
  ~RunHistogramManager();

  TH1F**  makeTH1F(const char* name, const char* title, const unsigned int nbinx, const double xmin, const double xmax);
  TProfile**  makeTProfile(const char* name, const char* title, const unsigned int nbinx, const double xmin, const double xmax);
  TH2F**  makeTH2F(const char* name, const char* title, const unsigned int nbinx, const double xmin, const double xmax, const unsigned int nbiny, const double ymin, const double ymax);
  TProfile2D**  makeTProfile2D(const char* name, const char* title, const unsigned int nbinx, const double xmin, const double xmax, const unsigned int nbiny, const double ymin, const double ymax);

  void  beginRun(const edm::Run& iRun);
  void  beginRun(const edm::Run& iRun, TFileDirectory& subdir);
  void  beginRun(const unsigned int irun);
  void  beginRun(const unsigned int irun, TFileDirectory& subdir);


 private:

  bool _fillHistograms;
  std::vector<BaseHistoParams*> _histograms;
  edm::EDGetTokenT<edm::ConditionsInRunBlock> _conditionsInRunToken;

};



#endif // DPGAnalysis_SiStripTools_RunHistogramManager_H
