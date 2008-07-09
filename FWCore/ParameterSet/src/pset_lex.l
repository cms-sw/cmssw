/*
 * $Id: pset_lex.l,v 1.33 2008/03/12 19:56:52 wmtan Exp $
 * Author: Us
 * Date: 4/28/05
 * History:
 *
 */


%{
#include <cstdio>
#include <cstring>
#include <string>

#include "FWCore/ParameterSet/interface/PSetNode.h"
#include "FWCore/ParameterSet/interface/VPSetNode.h"
#include "FWCore/ParameterSet/interface/EntryNode.h"
#include "FWCore/ParameterSet/interface/VEntryNode.h"

#include "FWCore/ParameterSet/src/pset_parse.h"
#include "FWCore/Utilities/interface/EDMException.h"

  using namespace edm::pset;

  namespace edm { namespace pset { extern int lines; } }
  char* stringDup(const char*);
  char* stringDupQuoted(const char* s,char quotetype);

  /* 
     YY_BREAK is #defined to nothing, to silence a C++ from
     complaining about unreachable code
  */

#define YY_BREAK
#define yylval pset_lval

  /* the SQ.DQ things are start states, or context dependent sections */
  /* DQ is used to identify a double-quoted string                    */
  /* SQ is used to identify a single-quoted string                    */


  /*
     If we remove the ? from the comment rules, and require the newline,
     our tests fail, complaining about a token #double.
     If we remove [\n]? from the comment rules, we obtain the same failure
     as when the [\n]? is there.
  */

%}


letterstart	[a-zA-Z]+[a-zA-Z0-9\-_]*
dotdelimited    ([a-zA-Z]+[a-zA-Z0-9\-_]*+\.)+[a-zA-Z]+[a-zA-Z0-9\-_]*
producttag      [a-zA-Z]+[a-zA-Z0-9\-_:]*
eventid         [0-9]*+:+[0-9]*
bangstart	\![a-zA-Z]+[a-zA-Z0-9\-_]*
minusinf        -inf
minusletterstart -[a-zA-Z]+[a-zA-Z0-9\-_]*
alphanumeric	[^\'\"\{\}\(\)\&=\n\t\r ,;:`]+
comment	        #.*[\n]?
comment2        "//".*[\n]?

%x DQ
%x SQ
%x CMNT

%%

"bool"		{ yylval.str=stringDup("bool"); return TYPE_tok; }
"int32"		{ yylval.str=stringDup("int32"); return TYPE_tok; }
"vint32"	{ yylval.str=stringDup("vint32"); return VTYPE_tok; }
"uint32"	{ yylval.str=stringDup("uint32"); return TYPE_tok; } 
"vuint32"	{ yylval.str=stringDup("vuint32"); return VTYPE_tok; } 
"int64"         { yylval.str=stringDup("int64"); return TYPE_tok; }
"vint64"        { yylval.str=stringDup("vint64"); return VTYPE_tok; }
"uint64"        { yylval.str=stringDup("uint64"); return TYPE_tok; }
"vuint64"       { yylval.str=stringDup("vuint64"); return VTYPE_tok; }
"string"	{ return STRING_tok; }
"vstring"	{ return VSTRING_tok; }
"double"	{ yylval.str=stringDup("double"); return TYPE_tok; }
"vdouble"	{ yylval.str=stringDup("vdouble"); return VTYPE_tok; }
"PSet"          { return PSET_tok; }
"VPSet"         { return VPSET_tok; }
"untracked"     { return UNTRACKED_tok; }
"FileInPath"    { return FILEINPATH_tok; }

"process"       { return PROCESS_tok; }
"source"        { yylval.str = stringDup(yytext); return SOURCE_tok; }
"looper"        { yylval.str = stringDup(yytext); return LOOPER_tok; }
"secsource"     { return SECSOURCE_tok; }
"es_source"     { yylval.str = stringDup(yytext); return ES_SOURCE_tok; }
"path"          { return PATH_tok; }
"sequence"      { return SEQUENCE_tok; }
"schedule"      { return SCHEDULE_tok; }
"block"         { return BLOCK_tok; }
"endpath"       { return ENDPATH_tok; }
"using"         { return USING_tok; }
"replace"       { return REPLACE_tok; }
"rename"        { return RENAME_tok; }
"copy"          { return COPY_tok; }
"include"       { return INCLUDE_tok; }
"from"          { return FROM_tok; }
"InputTag"      { return INPUTTAG_tok; }
"VInputTag"     { return VINPUTTAG_tok; }
"EventID"       { return EVENTID_tok; }
"VEventID"      { return VEVENTID_tok; }
"LuminosityBlockID" { return LUMINOSITYBLOCKID_tok; }
"VLuminosityBlockID" { return VLUMINOSITYBLOCKID_tok; }
"module"        { yylval.str = stringDup(yytext); return MODULE_tok; }
"service"       { yylval.str = stringDup(yytext); return SERVICE_tok; }
"es_module"     { yylval.str = stringDup(yytext); return ES_MODULE_tok; }
"es_prefer"     { yylval.str = stringDup(yytext); return ES_PREFER_tok; }
"mixer"         { return MIXER_tok; }
"mixer_path"    { return MIXERPATH_tok; }
"/*"            { BEGIN CMNT; break;}

\&              { return AND_tok; }
\(		{ return GROUP_START_tok; }
\)		{ return GROUP_END_tok; }

{letterstart}	{ yylval.str = stringDup(yytext); return LETTERSTART_tok; }
{dotdelimited}  { yylval.str = stringDup(yytext); return DOTDELIMITED_tok; }
{producttag}    { yylval.str = stringDup(yytext); return PRODUCTTAG_tok; }
{eventid}           { yylval.str = stringDup(yytext); return EVENTIDVALUE_tok;}
{bangstart}	{ yylval.str = stringDup(yytext); return BANGSTART_tok; }
{minusinf}      { yylval.str = stringDup(yytext); return MINUSINF_tok; }
{minusletterstart} {yylval.str = stringDup(yytext); return MINUSLETTERSTART_tok;}
{alphanumeric}	{ yylval.str = stringDup(yytext); return VALUE_tok; }
{comment}	{ ++lines; break ; }
{comment2}	{ ++lines; break ; }

\"\"		{ yylval.str = stringDupQuoted("",'"'); return DQWORD_tok; }
\'\'		{ yylval.str = stringDupQuoted("",'\''); return SQWORD_tok; }
\"		{ BEGIN DQ; break;}
\'		{ BEGIN SQ; break;}
\+=              { return PLUSEQUAL_tok; }
=		{ return EQUAL_tok; }
,               { return COMMA_tok; }
\{		{ return SCOPE_START_tok; }
\}		{ return SCOPE_END_tok; }
\n		{ ++lines; break; }
\r              { break ; }
\t              { break ; }
[ ]+		{ break ; }

<DQ>((\\\")*[^\"\n]*(\\\")*)+ { yylval.str = stringDupQuoted(yytext,'"'); return DQWORD_tok; }
<DQ>\"		{ BEGIN 0; break; }
<DQ>\n          { BEGIN 0; return ERROR_tok; }

<SQ>((\\\')*[^\'\n]*(\\\')*)+ { yylval.str = stringDupQuoted(yytext,'\''); return SQWORD_tok; }
<SQ>\'		{ BEGIN 0; break; }
<SQ>\n          { BEGIN 0; return ERROR_tok;}

<CMNT>. |
<CMNT>\n        { break; }
<CMNT>"*/"	{ BEGIN 0; break; }


%%

/* 
   The following was removed from the patterns above, 
   because it allowed some extraneous goo in files
   that should have been rejected
*/
#if 0
.		{ break ; }
#endif

int yywrap() { return 1; }


char* stringDup(const char* s)
{
  char* buffer = static_cast<char*>(malloc(strlen(s)+1));
  strcpy(buffer, s);
  return buffer;
}

char* stringDupQuoted(const char* s,char quotetype)
{
  int len = strlen(s);
  char* buffer = static_cast<char*>(malloc(len+1+2));
  strcpy(buffer+1, s);
  buffer[0]=quotetype;
  buffer[len+1]=quotetype;
  buffer[len+2]='\0';
  return buffer;
}


#include "FWCore/ParameterSet/interface/parse.h"
#include "boost/shared_ptr.hpp"

//  NOTE: the using declaration is before the include because the
//  header is autogenerated and I am not sure the autogenerator will
//  accept namespaces

using namespace edm::pset;
#include "FWCore/ParameterSet/src/pset_parse.h"

extern int pset_parse (void);

namespace edm {
   namespace pset {

     extern NodePtrList* global_gunk;

     std::string& errorMessage();
     // This is the function that actually creates the tree of nodes.
     // This function will either return a ParseResults object that
     // contains the parse results in "tree form", or will throw an
     // exception.

     ParseResults parse(char const* spec)
     {
       // Make sure state is clear.
       if (global_gunk)
	 throw edm::Exception(errors::Configuration, "BadParserState")
	   << "The configuration parser's global data is corrupt\n";
        
       // reset the error system
       errorMessage().clear();

       // Make the tokenizer read from the character buffer
       // rather than a FILE*.

       YY_BUFFER_STATE buf = yy_scan_string(spec);
       yy_switch_to_buffer(buf);
       lines=1;

       // Do the parsing; this sets global_gunk. Immediately grab
       // ownership of the created structure.
       if (pset_parse() != 0) {
	 yy_delete_buffer(buf);
         throw edm::Exception(errors::Configuration) 
	   << "Parsing of input file failed\n" 
           << errorMessage();
       }

       ParseResults ps(global_gunk);

       yy_delete_buffer(buf);

       // global_gunk is zero if there was an error
       if (!global_gunk) {
         //parser error?
         if(errorMessage().size()) {
            throw edm::Exception(errors::Configuration)<<errorMessage()<<"\n";
         }
         //something wierd must have happened.
	 throw edm::Exception(errors::Configuration, "BadParserState")
	   << "The configuration parser failed to set its global data\n";
       }
       errorMessage().clear();
       // ParseResults ps(global_gunk);
       // Don't delete global_gunk! The ParseResults object has taken
       // ownership.
       global_gunk=0;
       return ps;
     }
   }
}
