#include <iostream>
#include <iterator>
#include <algorithm>

#include "FWCore/ParameterSet/src/PythonFormWriter.h"
#include "FWCore/Utilities/interface/DebugMacros.h"

#include "FWCore/Utilities/interface/EDMException.h"


//
// TODO:
// 
//   We should clear the state at the beginning of write(), so that we
//   don't garble the result of a second use of the 'Writer. Or, we
//   should have the tree walk done during construction, and have
//   'write' just spew the guts. This would mean we need to make a new
//   'Writer for each parse tree we want to process.
//
//   There is too much replication of code in this class; there is
//   need of refactoring.
//
//   We can not deal with a PSet appearing a top level. There is no
//   place in the output format, as currently defined, so save such a
//   thing.

#define MYDEBUG(n) FDEBUG(n) << "DBG----- "

using namespace std;


namespace edm
{
  namespace pset
  {

    //------------------------------------------------------------
    // Helper functions

    typedef void (*writer_func)(ostream&, string const&);

    void 
    write_string_value(ostream& os, string const& val)
    {
      string val_without_trailing_quote(val, 0, val.size()-1);
      os << "'\\" 
	 << val_without_trailing_quote 
	 << "\\"             // escape for trailing quote
	 << *(val.rbegin())  // the trailing quote in string
	 << "'"; 
    }

    void 
    write_other_value(ostream& os, string const& val)
    {
      os << "'" << val << "'";
    }

    // For some reason, if the 'tracked' parameter is true, this means
    // the parameter is untracked.
    void
    write_trackedness(ostream& os, bool trackedval)
    {
      if (trackedval) 
	os << "'untracked'";
      else 
	os << "'tracked'";
    }

    // Cheesy heuristic for recognizing output modules from the class
    // name: If the name ends in OutputModule, we guess it is an
    // output module.

    bool
    looks_like_an_output_module(string const& classname)
    {
      string::size_type pos = classname.find("OutputModule");

      // If we didn't find OutputModule, it isn't an output module
      if ( pos == string::npos ) return false;

      // Now make sure some tricky lad didn't put OutputModule in the
      // middle of a class name... 
      //  length of OutputModule = 12

      return  (pos + 12) ==  classname.size();
    }
    

    //------------------------------------------------------------

    PythonFormWriter::PythonFormWriter() :
      procname_(),
      moduleStack_(),
      modules_(),
      processingVPSet_(false),
      nVPSetChildren_(0)
    {
      list<string> emptylist;
      modules_.insert(make_pair(string("es_module"), emptylist));
      modules_.insert(make_pair(string("es_source"), emptylist));
      modules_.insert(make_pair(string("es_prefer"), emptylist));
      modules_.insert(make_pair(string("module"), emptylist));
      modules_.insert(make_pair(string("source"), emptylist));
      modules_.insert(make_pair(string("sequence"),emptylist));
      modules_.insert(make_pair(string("path"),emptylist));
      modules_.insert(make_pair(string("endpath"),emptylist));
      modules_.insert(make_pair(string("service"),emptylist));
    }

    PythonFormWriter::~PythonFormWriter()
    { }

    void 
    PythonFormWriter::visitUsing(const UsingNode&)
    { 
      MYDEBUG(5) << "Saw a UsingNode\n";
    }

    void 
    PythonFormWriter::visitString(const StringNode&)
    { 
      MYDEBUG(5) << "Saw a StringNode\n";
    }


    // Entries withing modules that are not vectors or PSets.
    void 
    PythonFormWriter::visitEntry(const EntryNode& n)
    { 
      MYDEBUG(5) << "Saw an EntryNode\n";
      ostringstream tuple;

      tuple << "'"
	    << n.name << "': ('"
	    << n.type() << "', ";
      write_trackedness(tuple, n.tracked_);
      tuple << ", ";

      if (n.type() == "string") 
	{
	  write_string_value(tuple, n.value_);
	}
      else 
	{
	  write_other_value(tuple, n.value_);
	}

      tuple  << ')';
      
      moduleStack_.top() += tuple.str();
    }


    void
    PythonFormWriter::visitVEntry(const VEntryNode& n)
    { 
      MYDEBUG(5) << "Saw a VEntryNode\n";
      ostringstream tuple;

      tuple << "'"
	    << n.name << "': ('"
	    << n.type() << "', ";

      write_trackedness(tuple, n.tracked_);
      tuple << ", ";

      // Write out contents of the list...
      StringList::const_iterator i = n.value_->begin();
      StringList::const_iterator e = n.value_->end();

      // Figure out which writer to call, so we don't have to do it
      // each time in the loop below.
      writer_func writer_to_call = 
	( n.type() == "vstring" )  // not n.type() == "string"!
	? &write_string_value
	: &write_other_value;

      tuple << "[ ";
      for ( bool first = true; i != e; ++i, first = false)
	{
	  if (!first) tuple << ", ";
	  writer_to_call(tuple, *i);
	}

      tuple << " ])";
      
      moduleStack_.top() += tuple.str();
    }

    void
    PythonFormWriter::visitPSetRef(const PSetRefNode&)
    { 
      MYDEBUG(5) << "Saw a PSetRefNode: unimplemented\n";
    }

    void
    PythonFormWriter::visitContents(const ContentsNode& n)
    { 
      MYDEBUG(5) << "Saw a ContentsNode\n";

      // If the module stack is not empty, we're working on a PSet
      // inside a module (maybe inside something inside of a
      // module). Otherwise, we're working on a top-level PSet (not
      // currently working), or on the process block itself.
      if ( ! moduleStack_.empty() )
	{
         if(processingVPSet_ && nVPSetChildren_++) {
            //if this is actually a PSet embedded in a VPSet then we will need
            // to comma separate the children
            moduleStack_.top()+= ",";
         }
         
	  moduleStack_.top() += "{";

	  // We can't just call acceptForChildren, because we need to
	  // do something between children.
	  //
	  //n.acceptForChildren(*this);
	  NodePtrList::const_iterator i = n.value_->begin();
	  NodePtrList::const_iterator e = n.value_->end();
	  for ( bool first = true; i != e; first = false, ++i)
	    {
	      if (!first) moduleStack_.top() += ", ";
	      (*i)->accept(*this);	      
	    }

	  //moduleStack_.top() += "}\n";
	  moduleStack_.top() += "}";
	}
      else
	{
	  n.acceptForChildren(*this);
	}
    }

    void
    PythonFormWriter:: visitPSet(const PSetNode& n)
    { 
      MYDEBUG(5) << "Saw a PSetNode\n";
      if ( "process" == n.type() )
	{
	  procname_ = n.name;
	  n.acceptForChildren(*this);

	  MYDEBUG(4) << "\nprocess name: " << procname_
		     << "\nstack size? " << moduleStack_.size()
		     << "\nnumber of module types? " << modules_.size()
		     << '\n';
	}
      else if ( "PSet" == n.type() ) 
	{
	  // We're processing (the contents of) a PSet if we got
	  // here. The following processing assumes this PSet should
	  // be written a named parameter.
	  ostringstream out;
	  out << "'" 
	      << n.name 
	      << "': ('PSet', 'tracked', ";

	  moduleStack_.top() += out.str();

	  // Now print the guts...
	  n.acceptForChildren(*this);

	  // And finish up
	  //moduleStack_.top() += ")\n";
	  moduleStack_.top() += ")";
	}
      else
	{
	  MYDEBUG(5) << "weird thing: "
		     << n.type() << " " << n.name << '\n';
	}

    }

    void
    PythonFormWriter::visitVPSet(const VPSetNode& n)
    { 
      MYDEBUG(5) << "Saw a VPSetNode\n";
      ostringstream out;
      out << "'"
	  << n.name
	  << "': ('VPSet', 'tracked', [";
      moduleStack_.top() += out.str();

      
      moduleStack_.top() += "\n#start acceptForChildren in VPSetNode\n";
      processingVPSet_ = true;
      nVPSetChildren_=0;
      n.acceptForChildren(*this);
      processingVPSet_ = false;
      moduleStack_.top() += "\n#end acceptForChildren in VPSetNode\n";

      moduleStack_.top() += "])";
    }

    void
    PythonFormWriter::visitModule(const ModuleNode& n)
    { 
      MYDEBUG(5) << "Saw a ModuleNode, name: " 
		 << n.name << '\n';

      ostringstream header;

      // We don't want to write the name 'nameless' for unnamed
      // es_modules, nor the name 'main_es_input' for unnamed
      // es_sources, nor an empty string for the unnamed (main)
      // source.
      if ( (n.type() == "es_module") ||
	   (n.type() == "es_source") ||
           (n.type() == "es_prefer")   )
	{
          //es_* items are unique based on 'C++ class' and 'label'
          string prefix("");
          string label("");
          string name("@");
          if((n.type() == "es_module" && n.name!="nameless" ||
              n.type() == "es_source" && n.name!="main_es_input") ||
              n.type() == "es_prefer" && n.name!="nameless")
          {
             label = n.name;
             name += n.name;
          }
          if(n.type() =="es_prefer") {
            prefix = "esprefer_";
          }
	  header <<"'"<< prefix << n.class_ <<name<<"': { '@label': ('string','tracked', '" <<label<<"'), ";
	}
      else if (n.type() == "source" && n.name.empty())
	{
	  // no header to write...
	}
      else if(n.type()=="service") 
        {
          header<<"'"<<n.class_<<"': {";
        }
      else
	{
	  header << "'" << n.name << "': {";
	}
      header << "'@classname': ('string', 'tracked', '"
	     << n.class_
	     << "')";

      // Remember the names of modules that are output modules...  We
      // use a cheesy heuristic; see looks_like_an_output_module for
      // the details

//       assert ( looks_like_an_output_module ("AsciiOutputModule") );
//       assert ( looks_like_an_output_module ("PoolOutputModule") );
//       assert ( !looks_like_an_output_module ("NotOutputModuleX") );
//       assert ( !looks_like_an_output_module ("") );
//       assert ( !looks_like_an_output_module ("X") );

      if ( n.type() == "module" && 
	   looks_like_an_output_module(n.class_) )
	{
	  outputModuleNames_.push_back(n.name);
	}

      moduleStack_.push(header.str());

      // We can't just call 'acceptForChildren', beacuse we have to
      // take action between children.
      //n.acceptForChildren(*this);
     
      NodePtrList::const_iterator i(n.nodes_->begin());
      NodePtrList::const_iterator e(n.nodes_->end());

       for (  ; i!=e; ++i)
 	{
 	  // If we are processing a 'process' block, moduleStack_ will
 	  // be empty; if we're processing a module, or a pset, it
 	  // won't be. We hope we don't get here otherwise.
 	  //if (!moduleStack_.empty()) moduleStack_.top() += "\n,";
	  if (!moduleStack_.empty()) moduleStack_.top() += ", ";
 	  (*i)->accept(*this);
 	}

      moduleStack_.top() += '}'; // add trailer

      string section_label = n.type();
      modules_[section_label].push_back(moduleStack_.top());
      moduleStack_.pop();
    }


    // sequence, path, endpath, come in 'WrapperNode'
    void
    PythonFormWriter::visitWrapper(const WrapperNode& n)
    {
      ostringstream header;
      header<<"'"<<n.name<<"' : '";
      moduleStack_.push(header.str());
      
      //processes the node held by the wrapper
      n.wrapped_->accept(*this);

      moduleStack_.top()+="'";
      modules_[n.type_].push_back(moduleStack_.top());
      moduleStack_.pop();
      MYDEBUG(5) << "Saw a WrapperNode, name: "
		 << n.name << '\n';
    }
    
    void 
    PythonFormWriter::visitOperator(const OperatorNode& n)
    {
      moduleStack_.top()+="(";
      n.left_->accept(*this);
      moduleStack_.top()+=n.type_;
      n.right_->accept(*this);
      moduleStack_.top()+=")";
    }
    void 
    PythonFormWriter::visitOperand(const OperandNode& n)
    {
      moduleStack_.top()+=n.name;
    }


    void
    PythonFormWriter::write(ParseResults& parsetree, ostream& out)
    {
      // The 'list of nodes' must really be the root of a tree
      assert (parsetree->size() == 1);

      // Walk the tree, accumulating state.
      edm::pset::NodePtr root = parsetree->front();
      root->accept(*this);

      // Now write what we have found
      out << "{\n'procname': '"
	  << procname_
	  << "'\n";

      out << ", 'main_input': {\n";
      {
         list<string> const& input = modules_["source"];
         if(!input.empty()){
	    out << *(input.begin()) << '\n';
         }
         // print guts of main input here
      }
      //NOTE: no extra '}' added since it is added in the previous printing
      out << " # end of main_input\n";

      //------------------------------
      // Print real modules
      //------------------------------
      out << ", 'modules': {\n";
      {
	list<string> const& mods = modules_["module"];
	list<string>::const_iterator i = mods.begin();
	list<string>::const_iterator e = mods.end();
	for ( bool first = true ; i!=e; first=false, ++i)
	  {
	    out << "#--------------------\n";
	    if (!first) out << ',';
	    out << *i << '\n';
	  }
      }
      out << "} #end of modules\n";

      //------------------------------
      // Print es_modules
      //------------------------------
      out << "# es_modules\n";
      {
	out << ", 'es_modules': {\n";
	list<string> const& mods = modules_["es_module"];
	list<string>::const_iterator i = mods.begin();
	list<string>::const_iterator e = mods.end();
	for ( bool first = true; i!=e; first=false,++i)
	  {
	    if (!first) out << ',';
	    out << *i << '\n';
	  }
	out << "} #end of es_modules\n";
      }

      //------------------------------
      // Print es_sources
      //------------------------------
      out << "# es_sources\n";
      {
	out << ", 'es_sources': {\n";
	
	list<string> const& sources = modules_["es_source"];
	list<string>::const_iterator i = sources.begin();
	list<string>::const_iterator e = sources.end();
	for ( bool first = true; i!=e; first=false,++i)
	  {
	    if (!first) out << ',';
	    out << *i << '\n';
	  }
	cout << "} #end of es_sources\n";
      }

      //------------------------------
      // Print es_prefers
      //------------------------------
      out << "# es_prefers\n";
      {
	out << ", 'es_prefers': {\n";
	list<string> const& mods = modules_["es_prefer"];
	list<string>::const_iterator i = mods.begin();
	list<string>::const_iterator e = mods.end();
	for ( bool first = true; i!=e; first=false,++i)
        {
          if (!first) out << ',';
          out << *i << '\n';
        }
	out << "} #end of es_prefers\n";
      }

      out << "# output modules (names)\n";
      {
	out << ", 'output_modules': [ ";
	list<string>::const_iterator i = outputModuleNames_.begin();
	list<string>::const_iterator e = outputModuleNames_.end();
	for ( bool first = true; i !=e; first=false, ++i)
	  {
	    if (!first) out << ", ";
	    out << "'" << *i << "'";
	  }
	out << " ]\n" ;
      }
      out << "# sequences\n";
      {
         out <<", 'sequences': { \n";
	list<string> const& sources = modules_["sequence"];
	list<string>::const_iterator i = sources.begin();
	list<string>::const_iterator e = sources.end();
	for ( bool first = true; i!=e; first=false,++i)
        {
          if (!first) out << ',';
          out << *i << '\n';
        }
        out <<"}\n";
      }
      out << "# paths\n";
      {
         out <<", 'paths': { \n";
	list<string> const& sources = modules_["path"];
	list<string>::const_iterator i = sources.begin();
	list<string>::const_iterator e = sources.end();
	for ( bool first = true; i!=e; first=false,++i)
        {
          if (!first) out << ',';
          out << *i << '\n';
        }
        out <<"}\n";
      }
      out << "# endpaths\n";
      {
         out <<", 'endpaths': { \n";
	list<string> const& sources = modules_["endpath"];
	list<string>::const_iterator i = sources.begin();
	list<string>::const_iterator e = sources.end();
	for ( bool first = true; i!=e; first=false,++i)
        {
          if (!first) out << ',';
          out << *i << '\n';
        }
        out <<"}\n";
      }
      
      //------------------------------
      // Print services
      //------------------------------
      out << "# services\n";
      {
	out << ", 'services': {\n";
	list<string> const& mods = modules_["service"];
	list<string>::const_iterator i = mods.begin();
	list<string>::const_iterator e = mods.end();
	for ( bool first = true; i!=e; first=false,++i)
        {
          if (!first) out << ',';
          out << *i << '\n';
        }
	out << "} #end of es_modules\n";
      }
      
      out << '}';
    }

  } // namespace pset
} // namespace edm
