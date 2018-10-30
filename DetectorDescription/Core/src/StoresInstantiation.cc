#include "DetectorDescription/Core/interface/DDRotationMatrix.h"
#include "DetectorDescription/Core/interface/DDReadMapType.h"
#include "DetectorDescription/Core/interface/Singleton.h"
#include "DetectorDescription/Core/interface/Singleton.icc"
#include "DetectorDescription/Core/interface/Store.h"
#include "DetectorDescription/Core/interface/DDAxes.h"
#include "DetectorDescription/Core/interface/DDName.h"
#include "DetectorDescription/Core/interface/DDRoot.h"
#include "DetectorDescription/Core/src/Division.h"
#include "DetectorDescription/Core/src/LogicalPart.h"
#include "DetectorDescription/Core/src/Material.h"
#include "DetectorDescription/Core/src/Solid.h"
#include "DetectorDescription/Core/src/Specific.h"

#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

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
template class DDI::Singleton<std::map<std::pair<std::string, std::string>, int> >;
template class DDI::Singleton<std::map<std::string, std::vector<DDName> > >;
template class DDI::Singleton<std::vector<std::map<std::pair<std::string, std::string>, int>::const_iterator >  >;
