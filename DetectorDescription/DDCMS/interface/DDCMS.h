//==========================================================================
//  AIDA Detector description implementation 
//--------------------------------------------------------------------------
// Copyright (C) Organisation europeenne pour la Recherche nucleaire (CERN)
// All rights reserved.
//
// For the licensing terms see $DD4hepINSTALL/LICENSE.
// For the list of contributors see $DD4hepINSTALL/doc/CREDITS.
//
// Author     : M.Frank
//
//==========================================================================
//
// DDCMS is a detector description convention developed by the CMS experiment.
//
//==========================================================================
#ifndef DD4HEP_DDCMS_DDCMS_H
#define DD4HEP_DDCMS_DDCMS_H

// Framework includes
#include "XML/XML.h"
#include "DD4hep/DetElement.h"
#include "DetectorDescription/DDCMS/interface/DDCMSTags.h"

// C/C++ include files
#include <map>
#include <sstream>

/// Namespace for the AIDA detector description toolkit
namespace dd4hep {

  /// Namespace of DDCMS conversion namespace
  namespace cms  {

    // Forward declaration
    class ParsingContext;

    /// XML namespace for the evaluator
    /*
     *
     * \author  M.Frank
     * \version 1.0
     * \ingroup DD4HEP_CORE
     */
    class Namespace  {
    public:
      /// Reference to the global parsing context
      ParsingContext* context = 0;
      std::string     name;
      bool            pop = false;
    public:
      /// NO Default constructor
      Namespace() = delete;
      /// Initializing constructor
      Namespace(ParsingContext* ctx, xml_h element);
      Namespace(ParsingContext& ctx, xml_h element, bool);
      /// Initializing constructor
      Namespace(ParsingContext* ctx);
      /// Initializing constructor
      Namespace(ParsingContext& ctx);
      /// NO Copy constructor
      Namespace(const Namespace& copy) = delete;
      /// Standard destructor (Non virtual!)
      ~Namespace();
      /// NO assignment
      Namespace& operator=(const Namespace& copy) = delete;
      /// Prepend name with namespace
      std::string prepend(const std::string& n)  const;
      /// Resolve namespace during XML parsing
      std::string real_name(const std::string& v)  const;
      /// Strip off the namespace part of a given name
      static std::string obj_name(const std::string& name);
      /// Return the namespace name of a component
      static std::string ns_name(const std::string& n);
      template <typename T> T attr(xml_elt_t elt,const xml_tag_t& n) const   {
        std::string val = real_name(elt.attr<std::string>(n));
        elt.setAttr(n,val);
        return elt.attr<T>(n);
      }
      /// Add a new constant to the namespace
      void     addConstant(const std::string& name, const std::string& value, const std::string& type)  const;
      /// Add a new constant to the namespace as fully indicated by the name
      void     addConstantNS(const std::string& name, const std::string& value, const std::string& type)  const;

      /// Access material by its namespace dressed name
      Material material(const std::string& name)  const;
      Solid    solid(const std::string& name)  const;
      /// Add solid to current namespace
      Solid    addSolid(const std::string& name,Solid solid)  const;
      /// Add solid to current namespace as fully indicated by the name
      Solid    addSolidNS(const std::string& name,Solid solid)  const;

      Volume   volume(const std::string& name, bool exc=true)  const;
      /// Add volume to current namespace
      Volume   addVolume(Volume vol)  const;
      /// Add volume to current namespace as fully indicated by the name
      Volume   addVolumeNS(Volume vol)  const;

      const Rotation3D& rotation(const std::string& name)  const;
      /// Add rotation matrix to current namespace
      void    addRotation(const std::string& name,const Rotation3D& rot)  const;
    };

    /// XML parser context to store intermediate stuff
    /*
     *
     * \author  M.Frank
     * \version 1.0
     * \ingroup DD4HEP_CORE
     */
    class ParsingContext  {
    public:
      Detector*                          description;
      std::map<std::string, Rotation3D>  rotations;
      std::map<std::string, Solid>       shapes;
      std::map<std::string, Volume>      volumes;
      std::set<std::string>              disabledAlgs;
      std::vector<std::string>           namespaces;
      bool geo_inited = false;

      // Debug flags
      bool debug_includes     = false;
      bool debug_constants    = false;
      bool debug_materials    = false;
      bool debug_rotations    = false;
      bool debug_shapes       = false;
      bool debug_volumes      = false;
      bool debug_placements   = false;
      bool debug_namespaces   = false;
      bool debug_visattr      = false;
      bool debug_algorithms   = false;

    public:
      ParsingContext(Detector* det) : description(det) { namespaces.push_back(""); }
      ~ParsingContext() = default;
      const std::string& ns() const  {  return namespaces.back(); }
    };

    /// Encapsulation of the CMS detector description algorithm arguments
    /*
     *
     * \author  M.Frank
     * \version 1.0
     * \ingroup DD4HEP_CORE
     */
    class AlgoArguments   {
    protected:
      /// Access raw argument as a string by name
      xml_h       raw_arg(const std::string& name)  const;
      /// Access namespace resolved argument as a string by name
      std::string resolved_scalar_arg(const std::string& name)  const;

    public:
      std::string          name;
      ParsingContext&      context;
      xml_h                element;
      
      /// Initializing constructor
      AlgoArguments(ParsingContext& ctxt, xml_h algorithm);
      /// Default constructor
      AlgoArguments() = delete;
      /// Copy constructor
      AlgoArguments(const AlgoArguments& copy) = delete;
      /// Assignment operator
      AlgoArguments& operator=(const AlgoArguments& copy) = delete;
      /// Standatd destructor
      ~AlgoArguments() = default;

      /// Access value of rParent child node
      std::string parentName()  const;
      /// Access value of child'name from the xml element
      std::string childName()  const;
      /// Check the existence of an argument by name
      bool find(const std::string& name)  const;
      /// Access typed argument by name
      template<typename T> T value(const std::string& name)  const;
      /// Shortcut to access string arguments
      std::string str(const std::string& nam)  const;
      /// Shortcut to access double arguments
      double dble(const std::string& nam)  const;
      /// Shortcut to access integer arguments
      int integer(const std::string& nam)  const;
      /// Shortcut to access vector<double> arguments
      std::vector<double> vecDble(const std::string& nam)  const;
      /// Shortcut to access vector<int> arguments
      std::vector<int> vecInt(const std::string& nam)  const;
      /// Shortcut to access vector<int> arguments
      std::vector<std::string> vecStr(const std::string& nam)  const;
    };

    /// Re-direct debug messages to the DD4hep printer
    /*
     *
     * \author  M.Frank
     * \version 1.0
     * \ingroup DD4HEP_CORE
     */
    class LogDebug : public std::stringstream  {
    protected:
      std::string tag;
      int  level;
      bool pop = false;
    public:
      LogDebug() = delete;
      LogDebug(const LogDebug& copy) = delete;
      LogDebug& operator=(const LogDebug& copy) = delete;
      LogDebug(const std::string& tag_value, bool /* set_context */);
      LogDebug(const std::string& tag_value);
      ~LogDebug();
      static void setDebugAlgorithms(bool value);
    };

    /// Re-direct warning messages to the DD4hep printer
    /*
     *
     * \author  M.Frank
     * \version 1.0
     * \ingroup DD4HEP_CORE
     */
    class LogWarn : public LogDebug  {
    public:
      LogWarn() = delete;
      LogWarn(const LogWarn& copy) = delete;
      LogWarn& operator=(const LogWarn& copy) = delete;
      LogWarn(const std::string& tag_value);
    };


    /// Helper: Convert the name of a placed volume into a subdetector name
    std::string detElementName(PlacedVolume pv);
    /// Create 3D rotation matrix from angles.
    Rotation3D make_rotation3D(double thetaX, double phiX,
                               double thetaY, double phiY,
                               double thetaZ, double phiZ);

  }   /* End namespace cms      */
}     /* End namespace dd4hep   */

#endif /* DD4HEP_DDCMS_DDCMS_H  */
