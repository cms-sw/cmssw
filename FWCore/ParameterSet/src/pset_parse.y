

%{

/*
 * $Id: pset_parse.y,v 1.1 2005/05/29 02:29:55 wmtan Exp $
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

#include "FWCore/ParameterSet/interface/Nodes.h"
// our includes here

//
// Macro to handle debugging output
//
// #define EDM_VERBOSE_DEBUG 1
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

static int yyerror(char* msg);

int yylex();

namespace edm 
{ 
  namespace pset 
  {
    int lines=1;
    NodePtrList* global_gunk;
  }
}

inline string toString(char* arg) { string s(arg); free(arg); return s; }


%}

%union
{
  VPSetNode*  _VPSetNode;
  VEntryNode* _VEntryNode;
  PSetNode*   _PSetNode;
  EntryNode*  _EntryNode;
  NodePtrList* _NodePtrList;
  PSetNodePtrList* _PSetNodePtrList;
  Node* _Node;
  std::string*            _String;
  StringList*             _StringList;
  char*                   str;
  bool                    _bool;
}

%token TYPE_tok
%token LETTERSTART_tok
%token EQUAL_tok
%left COMMA_tok
%token VALUE_tok
%token SQWORD_tok
%token DQWORD_tok
%token WORD_tok

%token UNTRACKED_tok
%token UINT32_tok
%token INT32_tok
%token BOOL_tok
%token STRING_tok
%token DOUBLE_tok
%token VUINT32_tok
%token VINT32_tok
%token VSTRING_tok
%token VDOUBLE_tok
%token PSID_tok
%token PSNAME_tok
%token PSET_tok
%token VPSET_tok
%token TYPE_tok
%token VTYPE_tok

%token SCOPE_START_tok
%token SCOPE_END_tok

%left  AND_tok
%token SOURCE_tok
%token ES_SOURCE_tok
%token PATH_tok
%token SEQUENCE_tok
%token BLOCK_tok
%token ENDPATH_tok
%token USING_tok
%token MODULE_tok
%token ES_MODULE_tok
%token PROCESS_tok
%token GROUP_START_tok
%token GROUP_END_tok

%%

/* set global_gunk to be a NodePtrList pointer */
main:            process
                 {
                   DBPRINT("main: process");
                   global_gunk = $<_NodePtrList>1;
                   DBMADE("main");
                 }
               | nodes
                 {
                   DBPRINT("main: nodes");
                   global_gunk = $<_NodePtrList>1;
                   DBMADE("main");
                 }
               ;

/* return a NodePtrList pointer */
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

/* return a NodePtr pointer */
node:            untracked TYPE_tok LETTERSTART_tok EQUAL_tok any
                 {
                   DBPRINT("node: TYPE");
                   bool tr = $<_bool>1;
                   string type(toString($<str>2));
                   string name(toString($<str>3));
                   string value(toString($<str>5));
                   EntryNode* en(new EntryNode(type,name,value,tr,lines));
                   $<_Node>$ = en;
                 }
               | 
                 untracked VTYPE_tok LETTERSTART_tok EQUAL_tok array
                 {
                   DBPRINT("node: VTYPE");
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
                   DBPRINT("node: STRING");
                   bool tr = $<_bool>1;
                   string name(toString($<str>3));
                   string value(toString($<str>5));
                   EntryNode* en(new EntryNode("string",name,value,tr,lines));
                   $<_Node>$ = en;
                 }
               |
                 untracked VSTRING_tok LETTERSTART_tok EQUAL_tok strarray
                 {
                   DBPRINT("node: VSTRING");
                   string name(toString($<str>3));
                   StringListPtr value($<_StringList>5);
                   bool tr = $<_bool>1;
                   VEntryNode* en(new VEntryNode("vstring",name,value,tr,lines));
                   $<_Node>$ = en;
                 }
               |
                 USING_tok LETTERSTART_tok
                 {
                   DBPRINT("node: USING");
                   string name(toString($<str>2));
                   UsingNode* en(new UsingNode(name,lines));
                   $<_Node>$ = en;
                 }
               |
                 allpset
                 {
                   $<_Node>$ = $<_Node>1;
                 }
               ;

/* return a PSetNode pointer */
allpset:         PSET_tok LETTERSTART_tok EQUAL_tok scoped
                 {
                   DBPRINT("node: PSET");
                   string name(toString($<str>2));
                   NodePtrListPtr value($<_NodePtrList>4);
                   PSetNode* en(new PSetNode("PSet",name,value,lines));
                   $<_Node>$ = en;
                 }
               |
                 PSET_tok LETTERSTART_tok EQUAL_tok any
                 {
                   string name(toString($<str>2));
                   string value(toString($<str>4));
                   PSetRefNode* en(new PSetRefNode(name,value,lines));
                   $<_Node>$ = en;
                 }
               |
                 VPSET_tok LETTERSTART_tok EQUAL_tok SCOPE_START_tok nodesarray SCOPE_END_tok
                 {
                   DBPRINT("node: VPSET");
                   string name(toString($<str>2));
                   NodePtrListPtr value($<_NodePtrList>5);
                   VPSetNode* en(new VPSetNode("VPSet",name,value,lines));
                   $<_Node>$ = en;
                 }
               ;

/* Returns a NodePtrList pointer */
nodesarray:      nodesarray COMMA_tok scoped
                 {
                   NodePtrList* p = $<_NodePtrList>1;
                   NodePtrListPtr nodes($<_NodePtrList>3);
                   NodePtr n(new ContentsNode(nodes,lines));
                   p->push_back(n);
                   $<_NodePtrList>$ = p;
                 }
               |
                 nodesarray COMMA_tok any
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
                 any
                 {
                   string word(toString($<str>1));
                   NodePtr n(new StringNode(word,lines));
                   NodePtrList* p(new NodePtrList);
                   p->push_back(n);
                   $<_NodePtrList>$ = p;
                 }
               ;


/* Return a StringList pointer */
strarray:        SCOPE_START_tok  stranys SCOPE_END_tok
                 {
                   DBPRINT("strarray: not empty");
                   $<_StringList>$ = $<_StringList>2;
                 }
               |
                 SCOPE_START_tok SCOPE_END_tok
                 {
                   DBPRINT("strarray: empty");
                   $<_StringList>$ = new StringList();
                 }
               ;

/* Return a StringList pointer */
array:           SCOPE_START_tok  anys SCOPE_END_tok
                 {
                   DBPRINT("array: not empty");
                   $<_StringList>$ = $<_StringList>2;
                 }
               |
                 SCOPE_START_tok SCOPE_END_tok
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
                 {
                   DBPRINT("any: VALUE");
                   $<str>$ = $<str>1;
                 }
               |
                 LETTERSTART_tok
                 {
                   DBPRINT("any: LETTERSTART");
                   $<str>$ = $<str>1;
                 }
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

/* return a NodePtrList pointer */
process:         PROCESS_tok LETTERSTART_tok EQUAL_tok SCOPE_START_tok procnodes SCOPE_END_tok
                 {
	           DBPRINT("process: processnodes");
		   string name(toString($<str>2));
		   NodePtrListPtr nodes($<_NodePtrList>5);
		   NodePtr node(new PSetNode("process",name,nodes,lines));
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
procnode:        allpset
                 {
		   DBPRINT("procnode: PSET");
		   $<_Node>$ = $<_Node>1;
		 }
	       |
	         SOURCE_tok EQUAL_tok LETTERSTART_tok scoped
		 {
		   DBPRINT("procnode: initSOURCE");
		   string type(toString($<str>3));
		   NodePtrListPtr nodelist($<_NodePtrList>4);
		   ModuleNode* wn(new ModuleNode("source","main_input",type,nodelist,lines));
		   $<_Node>$ = wn;
		 }
	       |
	         ES_SOURCE_tok EQUAL_tok LETTERSTART_tok scoped
		 {
		   DBPRINT("procnode: initES_SOURCE");
		   string type(toString($<str>3));
		   NodePtrListPtr nodelist($<_NodePtrList>4);
		   ModuleNode* wn(new ModuleNode("es_source","main_es_input",type,nodelist,lines));
		   $<_Node>$ = wn;
		 }
	       |
	         SOURCE_tok LETTERSTART_tok EQUAL_tok LETTERSTART_tok scoped
		 {
		   DBPRINT("procnode: SOURCE");
		   string name(toString($<str>2));
		   string type(toString($<str>4));
		   NodePtrListPtr nodelist($<_NodePtrList>5);
		   ModuleNode* wn(new ModuleNode("source",name,type,nodelist,lines));
		   $<_Node>$ = wn;
		 }
	       |
	         ES_SOURCE_tok LETTERSTART_tok EQUAL_tok LETTERSTART_tok scoped
		 {
		   DBPRINT("procnode: SOURCE");
		   string name(toString($<str>2));
		   string type(toString($<str>4));
		   NodePtrListPtr nodelist($<_NodePtrList>5);
		   ModuleNode* wn(new ModuleNode("es_source",name,type,nodelist,lines));
		   $<_Node>$ = wn;
		 }
	       |
	         BLOCK_tok procinlinenodes
		 {
		   DBPRINT("procnode: BLOCK");
		   $<_Node>$ = $<_PSetNode>2;
		 }
	       |
	         MODULE_tok LETTERSTART_tok EQUAL_tok LETTERSTART_tok scoped
		 {
		   DBPRINT("procnode: MODULE");
		   string name(toString($<str>2));
		   string type(toString($<str>4));
		   NodePtrListPtr nodelist($<_NodePtrList>5);
		   ModuleNode* wn(new ModuleNode("module",name,type,nodelist,lines));
		   $<_Node>$ = wn;
		 }
	       |
	         ES_MODULE_tok LETTERSTART_tok EQUAL_tok LETTERSTART_tok scoped
		 {
		   DBPRINT("procnode: ES_MODULE");
		   string name(toString($<str>2));
		   string type(toString($<str>4));
		   NodePtrListPtr nodelist($<_NodePtrList>5);
		   ModuleNode* wn(new ModuleNode("es_module",name,type,nodelist,lines));
		   $<_Node>$ = wn;
		 }
	       |
	         ES_MODULE_tok EQUAL_tok LETTERSTART_tok scoped
		 {
		   DBPRINT("procnode: namelistES_MODULE");
		   string type(toString($<str>3));
		   NodePtrListPtr nodelist($<_NodePtrList>4);
		   ModuleNode* wn(new ModuleNode("es_module","nameless",type,nodelist,lines));
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
	         ENDPATH_tok LETTERSTART_tok EQUAL_tok SCOPE_START_tok pathexp SCOPE_END_tok
		 {
		   DBPRINT("procnode: ENDPATH");
		   string name(toString($<str>2));
		   NodePtr path($<_Node>5);
		   WrapperNode* wn(new WrapperNode("endpath",name,path,lines));
		   $<_Node>$ = wn;
		 }
	       ;

/* Returns a NodePtrList pointer */
scoped:          SCOPE_START_tok nodes SCOPE_END_tok
                 {
		   DBPRINT("scope: nodes");
		   $<_NodePtrList>$ = $<_NodePtrList>2;
		 }
	       |
	         SCOPE_START_tok SCOPE_END_tok
		 {
		   DBPRINT("scope: empty");
		   NodePtrList* nodelist(new NodePtrList);
		   $<_NodePtrList>$ = nodelist;
		 }
	       ;

/* Returns a Node pointer */
procinlinenodes: LETTERSTART_tok EQUAL_tok SCOPE_START_tok nodes SCOPE_END_tok
                 {
		   DBPRINT("procinlinenodes:NAME");
		   string name(toString($<str>1));
		   NodePtrListPtr value($<_NodePtrList>4);
		   PSetNode* en(new PSetNode("block",name,value,lines));
		   $<_Node>$ = en;
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
worker:          LETTERSTART_tok
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

%%

extern char *pset_text;
int yyerror(char* msg)
{
  cerr << "Parse error on line: " << lines << " token: " << pset_text << endl;
  cerr << "message: " << msg << endl;
  return 0;
}

