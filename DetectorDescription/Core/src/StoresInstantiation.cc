#include "DetectorDescription/Core/interface/DDRotationMatrix.h"
#include "DetectorDescription/Core/interface/DDReadMapType.h"
#include "DetectorDescription/Core/interface/Singleton.h"
#include "DetectorDescription/Core/interface/Singleton.icc"
#include "DetectorDescription/Core/interface/Store.h"
#include "DetectorDescription/Core/interface/DDAxes.h"
#include "DetectorDescription/Core/interface/DDName.h"
#include "DetectorDescription/Core/interface/DDRoot.h"
#include "DetectorDescription/Core/src/Division.h"
#include "DetectorDescription/Core/interface/LogicalPart.h"
#include "DetectorDescription/Core/interface/Material.h"
#include "DetectorDescription/Core/interface/Solid.h"
#include "DetectorDescription/Core/interface/Specific.h"

#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include <tbb/concurrent_vector.h>
#include <tbb/concurrent_unordered_map.h>

template class DDI::Singleton<AxesNames>;
template class DDI::Singleton<DDRoot>;
template class DDI::Singleton<DDI::Store<DDName, std::unique_ptr<std::vector<std::string> > > >;
template class DDI::Singleton<DDI::Store<DDName, std::unique_ptr<std::string> > >;
template class DDI::Singleton<DDI::Store<DDName, std::unique_ptr<DDI::Material> > >;
template class DDI::Singleton<DDI::Store<DDName, std::unique_ptr<ReadMapType<double> > > >;
template class DDI::Singleton<DDI::Store<DDName, std::unique_ptr<std::vector<double> > > >;
template class DDI::Singleton<DDI::Store<DDName, std::unique_ptr<DDI::Specific> > >;
template class DDI::Singleton<DDI::Store<DDName, std::unique_ptr<DDI::LogicalPart> > >;
template class DDI::Singleton<DDI::Store<DDName, std::unique_ptr<DDI::Solid> > >;
template class DDI::Singleton<DDI::Store<DDName, std::unique_ptr<double> > >;
template class DDI::Singleton<DDI::Store<DDName, std::unique_ptr<DDRotationMatrix> > >;
template class DDI::Singleton<DDI::Store<DDName, std::unique_ptr<DDI::Division>, std::unique_ptr<DDI::Division> > >;
template class DDI::Singleton<std::map<std::string, std::vector<DDName> > >;  //Used internally by DDLogicalPart
//The following are used by DDName
template class DDI::Singleton<DDName::Registry>;
template class DDI::Singleton<DDName::IdToName>;
