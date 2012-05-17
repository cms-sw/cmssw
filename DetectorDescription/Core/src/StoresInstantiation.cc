#include <DetectorDescription/Base/interface/Singleton.h>
#include <DetectorDescription/Base/interface/Singleton.icc>
#include <DetectorDescription/Base/interface/Store.h>
// #include <DetectorDescription/Base/interface/Store.icc>
#include <DetectorDescription/Core/interface/DDName.h>
#include <DetectorDescription/Core/interface/DDRoot.h>
#include <DetectorDescription/Core/interface/DDAxes.h>
#include <DetectorDescription/Base/interface/DDReadMapType.h>
#include <DetectorDescription/Core/interface/DDCompactViewImpl.h>
#include <DetectorDescription/ExprAlgo/interface/AlgoPos.h>
#include <DetectorDescription/ExprAlgo/interface/ExprEvalSingleton.h>
#include <DetectorDescription/Base/interface/DDRotationMatrix.h>
#include <DetectorDescription/Core/interface/DDPartSelection.h>
#include "Material.h"
#include "Solid.h"
#include "LogicalPart.h"
#include "Specific.h"
#include "Division.h"


#include <string>
#include <map>
#include <vector>

// template class DDI::Store<DDName, std::vector<std::string>* >;
// template class DDI::Store<DDName, std::string* >;
// template class DDI::Store<DDName, DDI::Material*>;
// template class DDI::Store<DDName, ReadMapType<double>* >;
// template class DDI::Store<DDName, std::vector<double>* >;
// template class DDI::Store<DDName, AlgoPos*>;
// template class DDI::Store<DDName, DDI::Specific*>;
// template class DDI::Store<DDName, DDI::LogicalPart*>;
// template class DDI::Store<DDName, DDI::Solid*>;
// template class DDI::Store<DDName, double*>;
// template class DDI::Store<DDName, DDRotationMatrix*>;
// template class DDI::Store<DDName, DDI::Division*, DDI::Division*>;


template class DDI::Singleton<AxesNames>;
template class DDI::Singleton<ClhepEvaluator>;
template class DDI::Singleton<DDRoot>;
template class DDI::Singleton<DDCompactViewImpl>;
template class DDI::Singleton<DDI::Store<DDName, std::vector<std::string>* > >;
template class DDI::Singleton<DDI::Store<DDName, std::string* > >;
template class DDI::Singleton<DDI::Store<DDName, DDI::Material*> >;
template class DDI::Singleton<DDI::Store<DDName, ReadMapType<double>* > >;
template class DDI::Singleton<DDI::Store<DDName, std::vector<double>* > >;
template class DDI::Singleton<DDI::Store<DDName, AlgoPos*> >;
template class DDI::Singleton<DDI::Store<DDName, DDI::Specific*> >;
template class DDI::Singleton<DDI::Store<DDName, DDI::LogicalPart*> >;
template class DDI::Singleton<DDI::Store<DDName, DDI::Solid*> >;
template class DDI::Singleton<DDI::Store<DDName, double*> >;
template class DDI::Singleton<DDI::Store<DDName, DDRotationMatrix*> >;
template class DDI::Singleton<DDI::Store<DDName, DDI::Division*, DDI::Division*> >;
template class DDI::Singleton<std::vector<std::pair<std::string, std::string> > >;
template class DDI::Singleton<std::map<std::string, int> >;
template class DDI::Singleton<std::map<std::pair<std::string, std::string>, int> >;
template class DDI::Singleton<std::map<std::string, std::vector<DDName> > >;
template class DDI::Singleton<std::vector<std::map<std::pair<std::string, std::string>, int>::const_iterator >  >;

//DDI::Singleton<DDI::Store<DDName, std::basic_string<char, std::char_traits<char>, std::allocator<char> >*, std::basic_string<char, std::char_traits<char>, std::allocator<char> >*> >

//DDI::Singleton<std::map<std::pair<std::basic_string<char, std::char_traits<char>, std::allocator<char> >, std::basic_string<char, std::char_traits<char>, std::allocator<char> > >, int, std::less<std::pair<std::basic_string<char, std::char_traits<char>, std::allocator<char> >, std::basic_string<char, std::char_traits<char>, std::allocator<char> > > >, std::allocator<std::pair<std::pair<std::basic_string<char, std::char_traits<char>, std::allocator<char> >, std::basic_string<char, std::char_traits<char>, std::allocator<char> > > const, int> > > >

//DDI::Singleton<std::map<std::pair<std::basic_string<char, std::char_traits<char>, std::allocator<char> >, std::basic_string<char, std::char_traits<char>, std::allocator<char> > >, int, std::less<std::pair<std::basic_string<char, std::char_traits<char>, std::allocator<char> >, std::basic_string<char, std::char_traits<char>, std::allocator<char> > > >, std::allocator<std::pair<std::pair<std::basic_string<char, std::char_traits<char>, std::allocator<char> >, std::basic_string<char, std::char_traits<char>, std::allocator<char> > > const, int> > > >

//DDI::Singleton<std::vector<std::_Rb_tree_const_iterator<std::pair<std::pair<std::basic_string<char, std::char_traits<char>, std::allocator<char> >, std::basic_string<char, std::char_traits<char>, std::allocator<char> > > const, int> >, std::allocator<std::_Rb_tree_const_iterator<std::pair<std::pair<std::basic_string<char, std::char_traits<char>, std::allocator<char> >, std::basic_string<char, std::char_traits<char>, std::allocator<char> > > const, int> > > > >

