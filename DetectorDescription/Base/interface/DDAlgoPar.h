#ifndef DDAlgoPar_h
#define DDAlgoPar_h

#include <map>
#include <vector>
#include <string>

/*! type for passing std::string-valued parameters to DDAlgo */
typedef std::map<std::string, std::vector<std::string> > parS_type;

/*! type for passing double-valued parameters to DDAlgo */
typedef std::map<std::string, std::vector<double> > parE_type;

#endif
