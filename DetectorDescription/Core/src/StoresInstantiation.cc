#include <DetectorDescription/Base/interface/DDRotationMatrix.h>
#include <DetectorDescription/Base/interface/Singleton.h>
#include <DetectorDescription/Base/interface/Store.h>
#include <map>
#include <string>
#include <utility>
#include <vector>

class AxesNames;
class ClhepEvaluator;
class DDName;
class DDRoot;
namespace DDI {
class Division;
class LogicalPart;
class Material;
class Solid;
class Specific;
}  // namespace DDI
template <class V> class ReadMapType;

template class DDI::Singleton<AxesNames>;
template class DDI::Singleton<ClhepEvaluator>;
template class DDI::Singleton<DDRoot>;
template class DDI::Singleton<DDI::Store<DDName, std::vector<std::string>* > >;
template class DDI::Singleton<DDI::Store<DDName, std::string* > >;
template class DDI::Singleton<DDI::Store<DDName, DDI::Material*> >;
template class DDI::Singleton<DDI::Store<DDName, ReadMapType<double>* > >;
template class DDI::Singleton<DDI::Store<DDName, std::vector<double>* > >;
template class DDI::Singleton<DDI::Store<DDName, DDI::Specific*> >;
template class DDI::Singleton<DDI::Store<DDName, DDI::LogicalPart*> >;
template class DDI::Singleton<DDI::Store<DDName, DDI::Solid*> >;
template class DDI::Singleton<DDI::Store<DDName, double*> >;
template class DDI::Singleton<DDI::Store<DDName, DDRotationMatrix*> >;
template class DDI::Singleton<DDI::Store<DDName, DDI::Division*, DDI::Division*> >;
template class DDI::Singleton<std::map<std::pair<std::string, std::string>, int> >;
template class DDI::Singleton<std::map<std::string, std::vector<DDName> > >;
template class DDI::Singleton<std::vector<std::map<std::pair<std::string, std::string>, int>::const_iterator >  >;
