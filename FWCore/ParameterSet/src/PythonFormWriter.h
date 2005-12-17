#ifndef FWCore_ParameterSet_PythonFormWriter_h
#define FWCore_ParameterSet_PythonFormWriter_h


//------------------------------------------------------------
// $Id:$
//
//
// PythonFormWriter defines a class that is to be used to walk the
// node tree produced by the configuration file parser, and to write
// that configuration tree in the Python exchange format, as
// documented in the file
// FWCore/Framework/ParameterSet/test/complete.pycfg
//
// CAVEATS:
//
//
//   1. It may be best to replace this whole thing with a Python
//   module that just uses the parse tree; such a module can be
//   written using boost::python. A more sensible interface to the
//   parse tree would be very useful if this is to be done.
//
//   2. Few if any of the transformation planned for configuration
//   files are currently done, or done at the right level (by
//   manipulating the parse tree itself, rather than being done during
//   walking of the tree). This means few such transformations are
//   supported by this class.
//
//------------------------------------------------------------
#include <map>
#include <sstream>
#include <stack>
#include <string>

#include "FWCore/ParameterSet/interface/Nodes.h"

namespace edm
{
  namespace pset
  {
    class PythonFormWriter : public Visitor
    {
    public:
      PythonFormWriter();

      // Virtual interface inherited from Visitor
      virtual ~PythonFormWriter();

      virtual void visitUsing(const UsingNode&);
      virtual void visitString(const StringNode&);
      virtual void visitEntry(const EntryNode&);
      virtual void visitVEntry(const VEntryNode&);
      virtual void visitPSetRef(const PSetRefNode&);
      virtual void visitContents(const ContentsNode&);
      virtual void visitPSet(const PSetNode&);
      virtual void visitVPSet(const VPSetNode&);
      virtual void visitModule(const ModuleNode&);
      virtual void visitWrapper(const WrapperNode&);
      

      // Function to be called from the 'outside', to walk the given
      // node tree and write the Python format of this configuration
      // to the given ostream.
      void write(ParseResults& parsetree, std::ostream& out);

    private:
      // Data accumulated while walking the tree.

      // Mapping type of module to printable contents
      typedef std::map<std::string, std::list<std::string> > ModuleCache;

      std::string             procname_;
      std::stack<std::string> moduleStack_;
      ModuleCache             modules_;
      std::list<std::string>  outputModuleNames_;

    }; // struct PythonFormWriter
  } // namespace pset
} // namespace edm

#endif
