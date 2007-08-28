

%{

/*
 * $Id: pset_parse.y,v 1.59 2007/05/11 22:55:05 rpw Exp $
 *
 * Author: Us
 * Date:   4/28/05


 Notes:
 The lists is the "V" nodes (arguments in the constructor) can 
 pass ownership to the node, and the node can hold by shared_ptr.
 the constructor can take an auto_ptr to the thing from the rule.

 If the various keywords were token "TYPE" and "VTYPE" 
 and had the value of the keyword, then we could reduce the number
 of almost redundent rules to only 2 or 3


 */


#include <cstdio>
#include <cstdlib>
#include <list>
#include <string>
#include <algorithm>
#include <iterator>
#include <sstream>
#include <vector>
#include <iostream>
#include <cassert>

#include "boost/shared_ptr.hpp"

#include "FWCore/ParameterSet/interface/PSetNode.h"
#include "FWCore/ParameterSet/interface/VPSetNode.h"
#include "FWCore/ParameterSet/interface/ReplaceNode.h"
#include "FWCore/ParameterSet/interface/EntryNode.h"
#include "FWCore/ParameterSet/interface/RenamedIncludeNode.h"
#include "FWCore/ParameterSet/interface/ImplicitIncludeNode.h"
#include "FWCore/ParameterSet/interface/VEntryNode.h"
#include "FWCore/ParameterSet/interface/ModuleNode.h"
#include "FWCore/ParameterSet/interface/WrapperNode.h"
#include "FWCore/ParameterSet/interface/OperatorNode.h"
#include "FWCore/ParameterSet/interface/OperandNode.h"
#include "FWCore/ParameterSet/interface/Nodes.h"

// our includes here

//
// Macro to handle debugging output
//
//#define EDM_VERBOSE_DEBUG 1
//#define YYERROR_VERBOSE 1

#if defined (EDM_VERBOSE_DEBUG)
  #define DBPRINT(R) std::cout << "Rule " << R << " detected" << std::endl
  #define DBDUMP(c) std::cout << "\tData is: " << c << std::endl
  #define DBMADE(x) std::cout << "\tMade a " << x << std::endl
#else
  #define DBPRINT(R) 
  #define DBDUMP(c) 
  #define DBMADE(x) 
#endif

using namespace std;
using namespace edm::pset;

/*
    yyerror is called from the parser to indicate an error.
    It calls the function errorMsg(), which prints to stderr; yyerror then
    returns 1.
*/

static int yyerror(char const* msg);

int yylex();

namespace edm 
{ 
  namespace pset 
  {
    int lines=1;
    NodePtrList* global_gunk;
    string currentFile = "";
  }
}

inline string toString(char* arg) { string s(arg); free(arg); return s; }


%}

%union
{
  edm::pset::VPSetNode*       _VPSetNode;
  edm::pset::VEntryNode*      _VEntryNode;
  edm::pset::PSetNode*        _PSetNode;
  edm::pset::EntryNode*       _EntryNode;
  edm::pset::NodePtrList*     _NodePtrList;
  edm::pset::Node*            _Node;
  std::string*                _String;
  edm::pset::StringList*      _StringList;
  char*                       str;
  bool                        _bool;
}

%token ERROR_tok
%token TYPE_tok
%token LETTERSTART_tok
%token DOTDELIMITED_tok
%token PRODUCTTAG_tok
%token BANGSTART_tok
%token MINUSLETTERSTART_tok
%token MINUSINF_tok
%token EQUAL_tok
%token PLUSEQUAL_tok
%left COMMA_tok
%token VALUE_tok
%token SQWORD_tok
%token DQWORD_tok
%token WORD_tok

%token UNTRACKED_tok
%token UINT32_tok
%token INT32_tok
%token UINT64_tok
%token INT64_tok
%token BOOL_tok
%token STRING_tok
%token DOUBLE_tok
%token VUINT32_tok
%token VINT32_tok
%token VUINT64_tok
%token VINT64_tok
%token VSTRING_tok

%token VDOUBLE_tok
%token PSID_tok
%token PSNAME_tok
%token PSET_tok
%token VPSET_tok
%token TYPE_tok
%token VTYPE_tok
%token FILEINPATH_tok

%token SCOPE_START_tok
%token SCOPE_END_tok

%left  AND_tok
%token SOURCE_tok
%token LOOPER_tok
%token SECSOURCE_tok
%token ES_SOURCE_tok
%token PATH_tok
%token SCHEDULE_tok
%token SEQUENCE_tok
%token BLOCK_tok
%token ENDPATH_tok
%token USING_tok
%token REPLACE_tok
%token RENAME_tok
%token COPY_tok
%token INCLUDE_tok
%token FROM_tok
%token INPUTTAG_tok
%token VINPUTTAG_tok
%token EVENTID_tok
%token VEVENTID_tok
%token EVENTIDVALUE_tok
%token MODULE_tok
%token SERVICE_tok
%token ES_MODULE_tok
%token ES_PREFER_tok
%token MIXER_tok
%token MIXERPATH_tok
%token PROCESS_tok
%token GROUP_START_tok
%token GROUP_END_tok

%%

/* set global_gunk to be a NodePtrList pointer */
main:            /*empty */
                 {
                   DBPRINT("main: empty");
                   NodePtrList* p(new NodePtrList);
                   global_gunk = p;
                 }
               | process
                 {
                   DBPRINT("main: process");
                   global_gunk = $<_NodePtrList>1;
                   DBMADE("main");
                 }
               | anylevelnodes
                 {
                   DBPRINT("main: anylevelnodes");
                   global_gunk = $<_NodePtrList>1;
                   DBMADE("main");
                 }
               ;

/* Returns a NodePtrList pointer */
anylevelnodes:   anylevelnodes anylevelnode
                 {
                   DBPRINT("anylevelnodes: anylevelnodes anylevelnode");
                   NodePtrList* p = $<_NodePtrList>1;
                   NodePtr node($<_Node>2);
                   p->push_back(node);
                   $<_NodePtrList>$ = p;
                 }
               |
                 anylevelnode
                 {
                   DBPRINT("anylevelnodes: anylevelnode");
                   NodePtr node($<_Node>1);
                   NodePtrList* p(new NodePtrList);
                   p->push_back(node);
                   $<_NodePtrList>$ = p;
                 }
               ;

anylevelnode:    toplevelnode
                 {
                   $<_Node>$ = $<_Node>1;
                 }
               |
                 lowlevelnode
                 {
                   $<_Node>$ = $<_Node>1;
                 }
               |
                 eitherlevelnode
                 {
                   $<_Node>$ = $<_Node>1;
                 }
               ;
 
/* Return a NodePtrList pointer */
nodes:           nodes node
                 {
                   DBPRINT("nodes: nodes node");
                   NodePtrList* p = $<_NodePtrList>1;
                   NodePtr node($<_Node>2);
                   p->push_back(node);
                   $<_NodePtrList>$ = p;
                 }
               |
                 node
                 {
                   DBPRINT("nodes: node");
                   assert($<_Node>1);
                   NodePtr node($<_Node>1);
                   assert(node != 0);
                   NodePtrList* p(new NodePtrList);
                   assert(p);
                   p->push_back(node);
                   assert (p->size() == 1);
                   $<_NodePtrList>$ = p;
                   DBMADE("nodes: node");
                 }
               ;

/* Return a NodePtr pointer to something that can exist inside a module or pset */
lowlevelnode:    untracked TYPE_tok LETTERSTART_tok EQUAL_tok any
                 {
                   DBPRINT("lowlevelnode: TYPE");
                   bool tr = $<_bool>1;
                   string type(toString($<str>2));
                   string name(toString($<str>3));
                   string value(toString($<str>5));
                   EntryNode* en(new EntryNode(type,name,value,tr,lines));
                   $<_Node>$ = en;
                 }
               | 
                 untracked VTYPE_tok LETTERSTART_tok EQUAL_tok possiblyblankarray
                 {
                   DBPRINT("lowlevelnode: VTYPE");
                   bool tr = $<_bool>1;
                   string type(toString($<str>2));
                   string name(toString($<str>3));
                   StringListPtr value($<_StringList>5);
                   VEntryNode* en(new VEntryNode(type,name,value,tr,lines));
                   $<_Node>$ = en;
                 }
               | 
                 untracked STRING_tok LETTERSTART_tok EQUAL_tok anyquote
                 {
                   DBPRINT("lowlevelnode: STRING");
                   bool tr = $<_bool>1;
                   string name(toString($<str>3));
                   string value(toString($<str>5));
                   EntryNode* en(new EntryNode("string",name,value,tr,lines));
                   $<_Node>$ = en;
                 }
               |
                 untracked VSTRING_tok LETTERSTART_tok EQUAL_tok possiblyblankstrarray
                 {
                   DBPRINT("lowlevelnode: VSTRING");
                   string name(toString($<str>3));
                   StringListPtr value($<_StringList>5);
                   bool tr = $<_bool>1;
                   VEntryNode* en(new VEntryNode("vstring",name,value,tr,lines));
                   $<_Node>$ = en;
                 }
               | 
                 untracked FILEINPATH_tok LETTERSTART_tok EQUAL_tok anyquote
                 {
                   DBPRINT("lowlevelnode: FILEINPATH");
                   bool tr = $<_bool>1;
                   string name(toString($<str>3));
                   string value(toString($<str>5));
                   EntryNode* en(new EntryNode("FileInPath",name,value,tr,lines));
                   $<_Node>$ = en;
                 }
               |
                 untracked INPUTTAG_tok LETTERSTART_tok EQUAL_tok anyproducttag
                 {
                   DBPRINT("lowlevelnode: INPUTTAG");
                   bool tr = $<_bool>1;
                   string name(toString($<str>3));
                   string value(toString($<str>5));
                   EntryNode* en(new EntryNode("InputTag",name,value,tr,lines));
                   $<_Node>$ = en;
                 }
               |
                untracked VINPUTTAG_tok LETTERSTART_tok EQUAL_tok producttagarray
                 {
                   DBPRINT("lowlevelnode: VINPUTTAG");
                   string name(toString($<str>3));
                   StringListPtr value($<_StringList>5);
                   bool tr = $<_bool>1;
                   VEntryNode* en(new VEntryNode("VInputTag",name,value,tr,lines));
                   $<_Node>$ = en;
                 }
               |
                 untracked EVENTID_tok LETTERSTART_tok EQUAL_tok EVENTIDVALUE_tok
                 {
                   DBPRINT("lowlevelnode: EVENTID");
                   bool tr = $<_bool>1;
                   string name(toString($<str>3));
                   string value(toString($<str>5));
                   EntryNode* en(new EntryNode("EventID",name,value,tr,lines));
                   $<_Node>$ = en;
                 }
               |
                untracked VEVENTID_tok LETTERSTART_tok EQUAL_tok possiblyblankeventidarray
                 {
                   DBPRINT("lowlevelnode: VEVENTID");
                   string name(toString($<str>3));
                   StringListPtr value($<_StringList>5);
                   bool tr = $<_bool>1;
                   VEntryNode* en(new VEntryNode("VEventID",name,value,tr,lines));
                   $<_Node>$ = en;
                 }
               |
                 USING_tok LETTERSTART_tok
                 {
                   DBPRINT("lowlevelnode: USING");
                   string name(toString($<str>2));
                   UsingNode* en(new UsingNode(name,lines));
                   $<_Node>$ = en;
                 }
               ;

node:            lowlevelnode
                 {
                   $<_Node>$ = $<_Node>1;
                 }
               |
                 eitherlevelnode
                 {
                   $<_Node>$ = $<_Node>1;
                 }
               ;

/* Nodes that can exist either at process level, or inside other blocks */
eitherlevelnode:    allpset
                 {
                   $<_Node>$ = $<_Node>1;
                 }
               |
                 INCLUDE_tok anyquote
                 {
                   DBPRINT("procnode: INCLUDE");
                   string name(toString($<str>2));
                   IncludeNode * wn(new IncludeNode("include", name, lines));
                   $<_Node>$ = wn;
                 }
               ;
 
/* Return a PSetNode pointer */
allpset:         untracked PSET_tok LETTERSTART_tok EQUAL_tok scoped
                 {
                   DBPRINT("node: PSET (scoped)");
		   bool tr = $<_bool>1;
                   string name(toString($<str>3));
                   NodePtrListPtr value($<_NodePtrList>5);
                   PSetNode* en(new PSetNode("PSet",name,value,tr,lines));
                   $<_Node>$ = en;
                 }
               |
                 untracked PSET_tok DOTDELIMITED_tok EQUAL_tok scoped
                 {
                   /* TEMPORARILY allow dots in names, for MessageLogger */
                   DBPRINT("node: PSET (scoped)");
                   bool tr = $<_bool>1;
                   string name(toString($<str>3));
                   NodePtrListPtr value($<_NodePtrList>5);
                   PSetNode* en(new PSetNode("PSet",name,value,tr,lines));
                   $<_Node>$ = en;
                 }
               |
                 untracked VPSET_tok LETTERSTART_tok EQUAL_tok SCOPE_START_tok nodesarray SCOPE_END_tok
                 {
                   DBPRINT("node: VPSET");
		   bool tr = $<_bool>1;
                   string name(toString($<str>3));
                   NodePtrListPtr value($<_NodePtrList>6);
                   VPSetNode* en(new VPSetNode("VPSet",name,value,tr,lines));
                   $<_Node>$ = en;
                 }
               |
                 untracked VPSET_tok LETTERSTART_tok EQUAL_tok SCOPE_START_tok SCOPE_END_tok
                 {
                   DBPRINT("node: VPSET (empty)");
                   bool tr = $<_bool>1;
                   string name(toString($<str>3));
                   NodePtrListPtr value(new NodePtrList());
                   VPSetNode* en(new VPSetNode("VPSet",name,value,tr,lines));
                   $<_Node>$ = en;
                 }
               |
                 MIXER_tok LETTERSTART_tok EQUAL_tok LETTERSTART_tok scoped
		 {
		   DBPRINT("node: MIXER");
                   string name(toString($<str>2));
                   string type(toString($<str>4));
                   NodePtrListPtr nodelist($<_NodePtrList>5);
                   ModuleNode* wn(new ModuleNode("mixer",name,type,nodelist,lines));
                   $<_Node>$ = wn;
		 }
	       |
                 SECSOURCE_tok LETTERSTART_tok EQUAL_tok LETTERSTART_tok scoped		 
	         {
                   DBPRINT("node: SECSOURCE");
                   string name(toString($<str>2));
                   string type(toString($<str>4));
                   NodePtrListPtr nodelist($<_NodePtrList>5);
                   ModuleNode* wn(new ModuleNode("secsource",name,type,nodelist,lines));
                   $<_Node>$ = wn;
		 }
	       |
	         MIXERPATH_tok LETTERSTART_tok EQUAL_tok SCOPE_START_tok pathseq SCOPE_END_tok
	         {
                   DBPRINT("procnode: PATH");
                   string name(toString($<str>2));
                   NodePtr path($<_Node>5);
                   WrapperNode* wn(new WrapperNode("mixer_path",name,path,lines));
                   $<_Node>$ = wn;
		 }
	       ;

/* Returns a NodePtrList pointer.  This is for explicit VPSets */
nodesarray:      nodesarray COMMA_tok scoped
                 {
                   NodePtrList* p = $<_NodePtrList>1;
                   NodePtrListPtr nodes($<_NodePtrList>3);
                   NodePtr n(new ContentsNode(nodes,lines));
                   p->push_back(n);
                   $<_NodePtrList>$ = p;
                 }
               |
                 nodesarray COMMA_tok LETTERSTART_tok
                 {
                   NodePtrList* p = $<_NodePtrList>1;
                   string word(toString($<str>3));
                   NodePtr n(new StringNode(word,lines));
                   p->push_back(n);
                   $<_NodePtrList>$ = p;
                 }
               |
                 scoped
                 {
                   NodePtrListPtr n($<_NodePtrList>1);
                   NodePtr newnode(new ContentsNode(n,lines));
                   NodePtrList* p(new NodePtrList);
                   p->push_back(newnode);
                   $<_NodePtrList>$ = p;
                 }
               |
                 LETTERSTART_tok
                 {
                   string word(toString($<str>1));
                   NodePtr n(new StringNode(word,lines));
                   NodePtrList* p(new NodePtrList);
                   p->push_back(n);
                   $<_NodePtrList>$ = p;
                 }
               ;

/* We need to be a little more explicit about what goes into
   a VPSet used in a replace statement, because we can't have
   any ambiguity: is p1 an InputTag or the name of a top-level PSet?
   Dot-delimited names of PSets aren't allowed just yet
*/

vpset:           vpset COMMA_tok scoped
                 {
                   NodePtrList* p = $<_NodePtrList>1;
                   NodePtrListPtr nodes($<_NodePtrList>3);
                   NodePtr n(new ContentsNode(nodes,lines));
                   p->push_back(n);
                   $<_NodePtrList>$ = p;
                 }
               |
                 scoped
                 {
                   NodePtrListPtr n($<_NodePtrList>1);
                   NodePtr newnode(new ContentsNode(n,lines));
                   NodePtrList* p(new NodePtrList);
                   p->push_back(newnode);
                   $<_NodePtrList>$ = p;
                 }
               ;


/* Return a StringList pointer */
anyarray:        array
               |
                 strarray
               |
                 blankarray
               |
                 eventidarray
/*               |
                 producttagarray
*/               ;

possiblyblankstrarray: strarray
               |
                 blankarray
               ;


strarray:        SCOPE_START_tok  stranys SCOPE_END_tok
                 {
                   DBPRINT("strarray: not empty");
                   $<_StringList>$ = $<_StringList>2;
                 }
               ;

/* Return a StringList pointer */
possiblyblankarray: array
               |
                 blankarray
               ;

array:           SCOPE_START_tok  anys SCOPE_END_tok
                 {
                   DBPRINT("array: not empty");
                   $<_StringList>$ = $<_StringList>2;
                 }
               ;

blankarray:      SCOPE_START_tok SCOPE_END_tok
                 {
                   DBPRINT("array: empty");
                   $<_StringList>$ = new StringList();
                 }
               ;

/* Return StringList pointer */
anys:            anys COMMA_tok any
                 {
                   DBPRINT("anys: anys COMMA_tok any");
                   StringList* p = $<_StringList>1;
                   string s(toString($<str>3));
                   p->push_back(s);
                   $<_StringList>$ = p;
                 }
               |
                 any
                 {
                   DBPRINT("anys: any");
                   string s(toString($<str>1));
                   StringList* p(new StringList);
                   p->push_back(s);
                   $<_StringList>$ = p;
                 }
               ;

/* Return StringList pointer */
stranys:         stranys COMMA_tok anyquote
                 {
                   DBPRINT("anys: anys COMMA_tok any");
                   StringList* p = $<_StringList>1;
                   string s(toString($<str>3));
                   p->push_back(s);
                   $<_StringList>$ = p;
                 }
               |
                 anyquote
                 {
                   DBPRINT("anys: any");
                   string s(toString($<str>1));
                   StringList* p(new StringList);
                   p->push_back(s);
                   $<_StringList>$ = p;
                 }
               ;

/* returns char* */
any:             VALUE_tok
               |
                 LETTERSTART_tok
               |
                 MINUSINF_tok
               |
                 PRODUCTTAG_tok
               ;

/* with or without a colon.  Some people even use the "source"
   keyword as their tag.  Whatever.  We'll allow it
 */
anyproducttag:   PRODUCTTAG_tok
                 {
                   DBPRINT("anyproducttag: PRODUCTTAG");
                   $<str>$ = $<str>1;
                 }
               |
                 LETTERSTART_tok
                 {
                   DBPRINT("anyproducttag: LETTERSTART");
                   $<str>$ = $<str>1;
                 }
               |
                 SOURCE_tok
               ;

replaceEntry:    VALUE_tok
               |
                 LETTERSTART_tok
               |
                 PRODUCTTAG_tok
               |
                 anyquote
               |
                 MINUSINF_tok
               |
                 EVENTIDVALUE_tok
               |
                 SOURCE_tok /* sometimes people use the word 'source' as an InputTag */
               ;

producttags:     producttags COMMA_tok anyproducttag
                 {
                   DBPRINT("producttag: producttag COMMA_tok producttag");
                   StringList* p = $<_StringList>1;
                   string s(toString($<str>3));
                   p->push_back(s);
                   $<_StringList>$ = p;
                 }
               |
                 anyproducttag
                 {
                   DBPRINT("producttags: producttag");
                   string s(toString($<str>1));
                   StringList* p(new StringList);
                   p->push_back(s);
                   $<_StringList>$ = p;
                 }
               ;

producttagarray: SCOPE_START_tok  producttags SCOPE_END_tok
                 {
                   DBPRINT("producttagarray: not empty");
                   $<_StringList>$ = $<_StringList>2;
                 }
               |
                 SCOPE_START_tok SCOPE_END_tok
                 {
                   DBPRINT("producttagarray: empty");
                   $<_StringList>$ = new StringList();
                 }
               ;

eventids:        eventids COMMA_tok EVENTIDVALUE_tok
                 {
                   DBPRINT("eventids comma");
                   StringList* p = $<_StringList>1;
                   string s(toString($<str>3));
                   p->push_back(s);
                   $<_StringList>$ = p;
                 }
               |
                 EVENTIDVALUE_tok
                 {
                   DBPRINT("eventids");
                   string s(toString($<str>1));
                   StringList* p(new StringList);
                   p->push_back(s);
                   $<_StringList>$ = p;
                 }
               ;

eventidarray: SCOPE_START_tok eventids SCOPE_END_tok
                 {
                   DBPRINT("eventids: not empty");
                   $<_StringList>$ = $<_StringList>2;
                 }
               ;

possiblyblankeventidarray : eventidarray
               |
                 blankarray
               ;

/* Returns a C-string */
anyquote:        SQWORD_tok
                 {
                   DBPRINT("anyquote: SQWORD");
                   $<str>$ = $<str>1;
                 }
               |
                 DQWORD_tok
                 {
                   DBPRINT("anyquote: DQWORD");
 DBPRINT($<str>1);
                   $<str>$ = $<str>1;
                 }
               ;

/* returns true or false */
/* 
   Note that the boolean value is the answer
   to the question: "Is the 'untracked' keyword present?".
   This is, admittedly, obscure.
*/
untracked:
                 {
                   $<_bool>$ = false;
                 }
               |
                 UNTRACKED_tok
                 {
                   $<_bool>$ = true;
                 }
               ;
	       
	       



/*---------------------------------------------------- */
/*      rules for process sections                     */
/*---------------------------------------------------- */

/* Return a NodePtrList pointer */
process:         PROCESS_tok LETTERSTART_tok EQUAL_tok SCOPE_START_tok procnodes SCOPE_END_tok
                 {
                   DBPRINT("process: processnodes");
                   string name(toString($<str>2));
                   NodePtrListPtr nodes($<_NodePtrList>5);
                   NodePtr node(new PSetNode("process",name,nodes, false, lines));
                   NodePtrList* p(new NodePtrList);
                   p->push_back(node);
                   $<_NodePtrList>$ = p;
                 }
               ;

/* Returns a NodePtrList pointer */
procnodes:       procnodes procnode
                 {
                   DBPRINT("procnodes: procnodes procnode");
                   NodePtrList* p = $<_NodePtrList>1;
                   NodePtr node($<_Node>2);
                   p->push_back(node);
                   $<_NodePtrList>$ = p;
                 }
               |
                 procnode
                 {
                   DBPRINT("procnodes: procnode");
                   NodePtr node($<_Node>1);
                   NodePtrList* p(new NodePtrList);
                   p->push_back(node);
                   $<_NodePtrList>$ = p;
                 }
               ;

/* Returns a Node pointer */
procnode:        eitherlevelnode 
                 {
                   DBPRINT("procnode: any level");
                   $<_Node>$ = $<_Node>1;
                 }
               |
                 toplevelnode
                 {
                   $<_Node>$ = $<_Node>1;
                 }
               ;

toplevelnode:    BLOCK_tok LETTERSTART_tok EQUAL_tok scoped
                 {
                   DBPRINT("procnode: BLOCK");
                   string name(toString($<str>2));
                   NodePtrListPtr value($<_NodePtrList>4);
                   PSetNode* en(new PSetNode("block",name,value, false, lines));
                   $<_Node>$ = en;
                 }
               |
                 REPLACE_tok DOTDELIMITED_tok EQUAL_tok replaceEntry
                 {
                   DBPRINT("procnode: REPLACEVALUE");
                   string name(toString($<str>2));
                   string value(toString($<str>4));
                   EntryNode * entry = new EntryNode("replace",name, value, false, lines);
                   NodePtr entryPtr(entry);
                   ReplaceNode* wn(new ReplaceNode("replace", name, entryPtr, false, lines));
                   $<_Node>$ = wn;
                 }
               |
                 REPLACE_tok DOTDELIMITED_tok PLUSEQUAL_tok replaceEntry
                 {
                   DBPRINT("procnode: APPENDVALUE");
                   string name(toString($<str>2));
                   string value(toString($<str>4));
                   EntryNode * entry = new EntryNode("replace",name, value, false, lines);
                   NodePtr entryPtr(entry);
                   ReplaceNode* wn(new ReplaceNode("replaceAppend", name, entryPtr, true, lines));
                   $<_Node>$ = wn;
                 }
               |
                 REPLACE_tok DOTDELIMITED_tok PLUSEQUAL_tok DOTDELIMITED_tok
                 {
                   DBPRINT("procnode: APPENDNODE");
                   string name(toString($<str>2));
                   string value(toString($<str>4));
                   EntryNode * entry = new EntryNode("dotdelimited",value, value, false, lines);
                   NodePtr entryPtr(entry);
                   ReplaceNode* wn(new ReplaceNode("replaceAppend", name, entryPtr, true, lines));
                   $<_Node>$ = wn;
                 }
               |
                 REPLACE_tok DOTDELIMITED_tok EQUAL_tok anyarray
                 {
                   DBPRINT("node: REPLACEARRAY");
                   string name(toString($<str>2));
                   StringListPtr value($<_StringList>4);
                   VEntryNode* en(new VEntryNode("replace",name,value,false,lines));
                   NodePtr entryPtr(en);
                   ReplaceNode* wn(new ReplaceNode("replace", name, entryPtr, false, lines));
                   $<_Node>$ = wn;
                 }
               |
                 REPLACE_tok DOTDELIMITED_tok PLUSEQUAL_tok anyarray
                 {
                   DBPRINT("node: APPENDARRAY");
                   string name(toString($<str>2));
                   StringListPtr value($<_StringList>4);
                   VEntryNode* en(new VEntryNode("replace",name,value, false,lines));
                   NodePtr entryPtr(en);
                   ReplaceNode* wn(new ReplaceNode("replaceAppend", name, entryPtr, true, lines));
                   $<_Node>$ = wn;
                 }
               |
                 REPLACE_tok DOTDELIMITED_tok EQUAL_tok nonblankscoped
                 {
                   DBPRINT("procnode:REPLACESCOPE");
                   string name(toString($<str>2));
                   NodePtrListPtr value($<_NodePtrList>4);
                   PSetNode* en(new PSetNode("replace", name, value, false, lines));
                   NodePtr psetPtr(en);
                   ReplaceNode* wn(new ReplaceNode("replace", name, psetPtr, false, lines));
                   $<_Node>$ = wn;
                 }
               |
                 REPLACE_tok LETTERSTART_tok EQUAL_tok nonblankscoped
                 {
                   DBPRINT("procnode:REPLACETOPLEVELPSET");
                   string name(toString($<str>2));
                   NodePtrListPtr value($<_NodePtrList>4);
                   PSetNode* en(new PSetNode("replace", name, value, false, lines));
                   NodePtr psetPtr(en);
                   ReplaceNode* wn(new ReplaceNode("replace", name, psetPtr, false, lines));
                   $<_Node>$ = wn;
                 }
               |
                 REPLACE_tok LETTERSTART_tok EQUAL_tok LETTERSTART_tok scoped
                 {
                   DBPRINT("procnode: REPLACEMODULE");
                   string name(toString($<str>2));
                   string type(toString($<str>4));
                   NodePtrListPtr nodelist($<_NodePtrList>5);
                   ModuleNode * moduleNode(new ModuleNode("replace",name,type,nodelist,lines));
                   NodePtr entryPtr(moduleNode);
                   ReplaceNode* wn(new ReplaceNode("replace", name, entryPtr, false, lines));
                   $<_Node>$ = wn;
                 }
               |
                 REPLACE_tok DOTDELIMITED_tok EQUAL_tok SCOPE_START_tok vpset SCOPE_END_tok
                 {
                   DBPRINT("procnode: REPLACE_VPSET");
                   string name(toString($<str>2));
                   NodePtrListPtr value($<_NodePtrList>5);
                   VPSetNode* en(new VPSetNode("VPSet",name,value,false,lines));
                   NodePtr vpsetNodePtr(en);
                   ReplaceNode* wn(new ReplaceNode("replace", name, vpsetNodePtr, false, lines));
                   $<_Node>$ = wn;
                 }
               |
                 REPLACE_tok DOTDELIMITED_tok PLUSEQUAL_tok nonblankscoped
                 {
                   // need to add a rule for appending with a dot-delimited PSet
                   DBPRINT("procnode: SINGLE_APPEND_VPSET");
                   string name(toString($<str>2));
                   NodePtrListPtr value($<_NodePtrList>4);
                   NodePtr contentsPtr(new ContentsNode(value, lines));
                   ReplaceNode* wn(new ReplaceNode("replaceAppend", name, contentsPtr, true, lines));
                   $<_Node>$ = wn;
                 }
               |
                 REPLACE_tok DOTDELIMITED_tok PLUSEQUAL_tok SCOPE_START_tok vpset SCOPE_END_tok
                 {
                   DBPRINT("procnode: MULTIPLE_APPEND_VPSET");
                   string name(toString($<str>2));
                   NodePtrListPtr value($<_NodePtrList>5);
                   VPSetNode* en(new VPSetNode("VPSet",name,value,false,lines));
                   NodePtr vpsetNodePtr(en);
                   ReplaceNode* wn(new ReplaceNode("replaceAppend", name, vpsetNodePtr, true, lines));
                   $<_Node>$ = wn;
                 }
               |
                 RENAME_tok LETTERSTART_tok LETTERSTART_tok
                 {
                   DBPRINT("procnode: RENAME");
                   string from(toString($<str>2));
                   string to(toString($<str>3));
                   RenameNode * wn(new RenameNode("rename", from, to, lines));
                   $<_Node>$ = wn;
                 }
               |
/*                 COPY_tok LETTERSTART_tok LETTERSTART_tok
                 {
                   DBPRINT("procnode: COPY");
                   string from(toString($<str>2));
                   string to(toString($<str>3));
                   CopyNode * wn(new CopyNode("copy", from, to, lines));
                   $<_Node>$ = wn;
                 }
               | */
                 namedmodule LETTERSTART_tok EQUAL_tok LETTERSTART_tok scoped
                 {
                   DBPRINT("procnode: MODULE");
                   string type(toString($<str>1));
                   string name(toString($<str>2));
                   string classname(toString($<str>4));
                   NodePtrListPtr nodelist($<_NodePtrList>5);
                   ModuleNode* wn(new ModuleNode(type,name,classname,nodelist,lines));
                   $<_Node>$ = wn;
                 } 
               |
                 namedmodule SOURCE_tok EQUAL_tok LETTERSTART_tok scoped
                 {
                   DBPRINT("procnode: MODULE");
                   string type(toString($<str>1));
                   string name(toString($<str>2));
                   string classname(toString($<str>4));
                   NodePtrListPtr nodelist($<_NodePtrList>5);
                   ModuleNode* wn(new ModuleNode(type,name,classname,nodelist,lines));
                   $<_Node>$ = wn;
                 }
               |
                 unnamedmodule EQUAL_tok LETTERSTART_tok scoped
                 {
                   DBPRINT("procnode: SERVICE");
                   string type(toString($<str>1));
                   string classname(toString($<str>3));
                   NodePtrListPtr nodelist($<_NodePtrList>4);
                   ModuleNode* wn(new ModuleNode(type, "",classname,nodelist,lines));
                   $<_Node>$ = wn;
                 }
               |
                 namedmodule LETTERSTART_tok EQUAL_tok LETTERSTART_tok
                 {
                   DBPRINT("procnode: IMPLICITINCLUDE_MODULE");
                   string type(toString($<str>1));
                   string label(toString($<str>2));
                   string classname(toString($<str>4));
                   ImplicitIncludeNode* wn(new ImplicitIncludeNode(classname, label, lines));
                   $<_Node>$ = wn;
                 }
               |
                 unnamedmodule EQUAL_tok LETTERSTART_tok
                 {
                   DBPRINT("procnode: UNNAMED IMPLICITINCLUDE_MODULE");
                   string type(toString($<str>1));
                   string classname(toString($<str>3));
                   ImplicitIncludeNode* wn(new ImplicitIncludeNode(classname, classname, lines));
                   $<_Node>$ = wn;
                 }
               |
                 namedmodule LETTERSTART_tok EQUAL_tok LETTERSTART_tok FROM_tok anyquote
                 {
                   DBPRINT("procnode: RENAMEDINCLUDE");
                   string targetType(toString($<str>1));
                   string newName(toString($<str>2));
                   string targetName(toString($<str>4));
                   string includeFile(toString($<str>6));
                   RenamedIncludeNode * wn(new RenamedIncludeNode("includeRenamed", includeFile, targetType, newName, targetName, lines));
                   $<_Node>$ = wn;
                 }
               |
                 SEQUENCE_tok LETTERSTART_tok EQUAL_tok SCOPE_START_tok pathexp SCOPE_END_tok
                 {
                   DBPRINT("procnode: SEQ");
                   string name(toString($<str>2));
                   NodePtr path($<_Node>5);
                   WrapperNode* wn(new WrapperNode("sequence",name,path,lines));
                   $<_Node>$ = wn;
                 }
               |
                 PATH_tok LETTERSTART_tok EQUAL_tok SCOPE_START_tok pathexp SCOPE_END_tok
                 {
                   DBPRINT("procnode: PATH");
                   string name(toString($<str>2));
                   NodePtr path($<_Node>5);
                   WrapperNode* wn(new WrapperNode("path",name,path,lines));
                   $<_Node>$ = wn;
                 }
               |
                 SCHEDULE_tok EQUAL_tok SCOPE_START_tok pathexp SCOPE_END_tok
                 {
                   DBPRINT("procnode: SCHEDULE");
                   NodePtr path($<_Node>4);
                   WrapperNode* wn(new WrapperNode("schedule", "" ,path,lines));
                   $<_Node>$ = wn;
                 }
               |
                 ENDPATH_tok LETTERSTART_tok EQUAL_tok SCOPE_START_tok pathexp SCOPE_END_tok
                 {
                   DBPRINT("procnode: ENDPATH");
                   string name(toString($<str>2));
                   NodePtr path($<_Node>5);
                   WrapperNode* wn(new WrapperNode("endpath",name,path,lines));
                   $<_Node>$ = wn;
                 }
               ;
 
namedmodule:     MODULE_tok
               |
                 ES_MODULE_tok
               |
                 ES_PREFER_tok
               |
                 ES_SOURCE_tok
               ;

unnamedmodule:   SERVICE_tok
               |
                 ES_MODULE_tok
               |
                 ES_PREFER_tok
               |
                 SOURCE_tok
               |
                 ES_SOURCE_tok
               |
                 LOOPER_tok
               ;

/* Returns a NodePtrList pointer */
scoped:          nonblankscoped
               |
                 blankscoped;
               ;
 
nonblankscoped:  SCOPE_START_tok nodes SCOPE_END_tok
                 {
                   DBPRINT("scope: nodes");
                   $<_NodePtrList>$ = $<_NodePtrList>2;
                 }
               ;

blankscoped:     SCOPE_START_tok SCOPE_END_tok
                 {
                   DBPRINT("scope: empty");
                   NodePtrList* nodelist(new NodePtrList);
                   $<_NodePtrList>$ = nodelist;
                 }
               ;

/* Returns a Node pointer */
pathexp:         pathexp AND_tok pathseq
                 {
                   DBPRINT("pathexp: AND");
                   NodePtr nl($<_Node>1);
                   NodePtr nr($<_Node>3);
                   Node* op(new OperatorNode("&",nl,nr,lines));
                   $<_Node>$ = op;
                 }
               |
                 pathseq
                 {
                   DBPRINT("pathexp: pathseq");
                   $<_Node>$ = $<_Node>1;
                 }
               ;

/* Returns a Node pointer */
pathseq:         pathseq COMMA_tok worker
                 {
                   DBPRINT("pathseq: COMMA");
                   NodePtr nl($<_Node>1);
                   NodePtr nr($<_Node>3);
                   Node* op(new OperatorNode(",",nl,nr,lines));
                   $<_Node>$ = op;
                 }
               |
                 worker
                 {
                   DBPRINT("pathseq: worker");
                   $<_Node>$ = $<_Node>1;
                 }
               ;

/* Returns a Node pointer */
worker:          bangorletter
                 {
                   DBPRINT("worker: NAME");
                   string name(toString($<str>1));
                   OperandNode* op(new OperandNode("operand",name,lines));
                   $<_Node>$ = op;
                 }
               |
                 GROUP_START_tok pathexp GROUP_END_tok
                 {
                   DBPRINT("worker: grouppath");
                   $<_Node>$ = $<_Node>2;
                 }
               ;

bangorletter: LETTERSTART_tok
      { $<str>$=$<str>1 }
      | BANGSTART_tok
      { $<str>$=$<str>1 }
      | MINUSLETTERSTART_tok
      { $<str>$=$<str>1 }
      | SOURCE_tok
      { $<str>$=$<str>1 }
	  ;

%%


namespace edm {
   namespace pset {
      std::string& errorMessage()
      {
         static std::string s_message;
         return s_message;
      }
   }
}

extern char *pset_text;
int yyerror(char const* msg)
{
  std::stringstream err;
  err << "Parse error ";
   if(currentFile != "")
  {
    err << "in file " << currentFile << "\n";
  }
  err << "on line: " << lines << " token: '" << pset_text << "'\n";
  err << "message: " << msg << "\n";
  errorMessage() = err.str();
  return 0;
}
