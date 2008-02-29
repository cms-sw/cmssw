// MCDB API: parser_xs.cpp
// LCG MCDB project, Monte Carlo Data Base
// http://mcdb.cern.ch
// 
// Gene Galkin, IHEP, Protvino, Russia
// e-mail: galkine@ihep.ru, 2006-2007
//
// Sergey Belov <Sergey.Belov@cern.ch>, 2007
//

#include <iostream>
#include "GeneratorInterface/LHEInterface/src/parser_xs.h"


namespace mcdb {

namespace parser_xs {

XERCES_CPP_NAMESPACE_USE

class XStr
{
 public:
    XStr(const char* const toTranscode)
    {
        // Call the private transcoding method
        fUnicodeForm = XMLString::transcode(toTranscode);
    }
    
    ~XStr()
    {
        XMLString::release(&fUnicodeForm);
    }
    
    const XMLCh* unicodeForm() const
    {
        return fUnicodeForm;
    }
 private:
    //  This is the Unicode XMLCh format of the string.
    XMLCh* fUnicodeForm;
};

#define X(str) XStr::XStr(str).unicodeForm()

int init_parser_xs(DOMBuilder*& parser, bool reportErrors=false, bool processNs=false);
void finish_parser_xs(DOMBuilder*& parser);
DOMDocument* parse_document_xs(const string& uri, DOMBuilder* parser, int& error_code);
const Article get_article_xs(DOMDocument* doc);
const vector<File> get_files_xs(DOMDocument* doc);


HepmlParserXs::HepmlParserXs(): HepmlParser()
{}

HepmlParserXs::~HepmlParserXs()
{}


const Article HepmlParserXs::getArticle(const string& uri)
{
    DOMBuilder*  parser = 0;
    DOMDocument* doc    = 0;
    Article      article;
    int          error_code;
    
    using namespace std;
    
    error_code = init_parser_xs( parser,reportErrors(),processNs() );
    errorCode(error_code);
    if (error_code) return article;
    
    doc = parse_document_xs(uri,parser,error_code);
    errorCode(error_code);
    if (error_code) return article;
    
    article = get_article_xs( doc );
    finish_parser_xs(parser);

    errorCode(error_code);
    return article;
}

const vector<File> HepmlParserXs::getFiles(const string& uri)
{
    DOMBuilder* parser = 0;
    vector<File> files;
    int		error_code;
    
    error_code = init_parser_xs( parser,reportErrors(),processNs() );
    files = get_files_xs( parse_document_xs(uri,parser,error_code) );
    finish_parser_xs(parser);

    return files;
}




int init_parser_xs(DOMBuilder*& parser, bool reportErrors, bool processNs)
{
    // Initialize the Xerces XML system
    try
    {
        XMLPlatformUtils::Initialize();
    }
    catch(const XMLException& toCatch)
    {
        char* tchar = XMLString::transcode(toCatch.getMessage());
        std::cerr << "\nParser(): Error during initialization! Message:" +
           string(tchar) + string("\n");
        XMLString::release(&tchar);
        return 1;
    }

    static const XMLCh gLS[] = {chLatin_L, chLatin_S, chNull};
 
    DOMImplementation *impl =
        DOMImplementationRegistry::getDOMImplementation(gLS);
    parser = ((DOMImplementationLS*)impl)->createDOMBuilder(DOMImplementationLS::MODE_SYNCHRONOUS, 0);
    if (parser->canSetFeature(XMLUni::fgDOMNamespaces, processNs))
        parser->setFeature(XMLUni::fgDOMNamespaces, processNs);
    if (parser->canSetFeature(XMLUni::fgXercesSchema, true))
        parser->setFeature(XMLUni::fgXercesSchema, true);
    if (parser->canSetFeature(XMLUni::fgXercesSchemaFullChecking, true))
        parser->setFeature(XMLUni::fgXercesSchemaFullChecking, true);
 
 // ???
    if ( parser->canSetFeature(XMLUni::fgDOMValidation, reportErrors) )
        parser->setFeature(XMLUni::fgDOMValidation, reportErrors);
    if ( parser->canSetFeature(XMLUni::fgDOMValidateIfSchema, reportErrors ) )
        parser->setFeature(XMLUni::fgDOMValidateIfSchema, reportErrors);
    // datatype normalization - default is off
    if ( parser->canSetFeature(XMLUni::fgDOMDatatypeNormalization, true) )
        parser->setFeature(XMLUni::fgDOMDatatypeNormalization, true);
    
    return 0;
    
} // init_parser_xs
    

DOMDocument* parse_document_xs(const string& uri, DOMBuilder* parser, int& error_code)
{
    // create our error handler and install it
    using std::endl;
    
    mcdbDOMErrorHandler errorHandler;
    parser->setErrorHandler(&errorHandler);
    errorHandler.resetErrors();
    DOMDocument* doc = 0;
    DOMDocument* empty_doc = 0;
    
    error_code = 0;
    
    try
    {
        parser->resetDocumentPool();
        doc = parser->parseURI(uri.c_str());
    }
    catch(const XMLException& toCatch)
    {
        std::cerr << endl << "Parser(): Exception message is: " << endl;
        std::cerr << XMLString::transcode(toCatch.getMessage()) << endl;
        error_code = 4;
        return empty_doc;
    }
    catch(const DOMException& toCatch)
    {
        std::cerr << "\nParser(): Exception message is: \n";
        std::cerr << string(XMLString::transcode(toCatch.msg)) + string("\n");
        error_code = 4;
        return empty_doc;
    }
    catch(...)
    {
        std::cerr << "Unexpected Exception" << endl;
	error_code = 3;
        return empty_doc;
    }

    if(errorHandler.getSawErrors())
    {
        std::cerr << "Unexpected Error \n";
        error_code = 5;
        return empty_doc;
    }

    return doc;
}


void finish_parser_xs(DOMBuilder*& parser)
{
  // Delete the parser itself. Must be done prior to calling Terminate.
    parser->release();
    XMLPlatformUtils::Terminate();
} // finish_parser_xs


void file_node_xs(DOMNode* pnode, vector<File>& vfile)
{
 DOMNode* node;
 int i = vfile.size();
 vfile.resize(i+1);
 vfile[i].id(atoi(XMLString::transcode(pnode->getAttributes()->item(0)->getNodeValue())));
 string type = string(XMLString::transcode(pnode->getAttributes()->item(1)->getNodeValue()));
 
 if ( type == "data") vfile[i].type(data);
 else if (type == "metadata") vfile[i].type(metadata);
 else if (type == "generator_config") vfile[i].type(generator_config);
 else if (type == "generator_binary") vfile[i].type(generator_binary);
 else vfile[i].type(any);
 
 
 DOMNode* leaf = pnode->getFirstChild();
 while(leaf)
 {
  if(leaf->hasChildNodes())
  {
   node = leaf->getFirstChild();
   if(node->getNextSibling()) node = node->getNextSibling();
   if(string(XMLString::transcode(leaf->getNodeName())) == "eventsNumber")
    vfile[i].eventsNumber(atoi(XMLString::transcode(node->getNodeValue())));
   else if(string(XMLString::transcode(leaf->getNodeName())) ==
    "crossSectionPb")
    vfile[i].crossSectionPb(atof(XMLString::transcode(node->getNodeValue())));
   else if(string(XMLString::transcode(leaf->getNodeName())) =="csErrorPlusPb")
    vfile[i].csErrorPlusPb(atof(XMLString::transcode(node->getNodeValue())));
   else if(string(XMLString::transcode(leaf->getNodeName())) == "csErrorMinusPb")
    vfile[i].csErrorMinusPb(atof(XMLString::transcode(node->getNodeValue())));
   else if(string(XMLString::transcode(leaf->getNodeName())) == "fileSize")
    vfile[i].size(atol(XMLString::transcode(node->getNodeValue())));
   else if(string(XMLString::transcode(leaf->getNodeName())) == "checksum")
    vfile[i].checksum(XMLString::transcode(node->getNodeValue()));
   else if(string(XMLString::transcode(leaf->getNodeName())) == "comments")
    vfile[i].comments(XMLString::transcode(node->getNodeValue()));
   else if(string(XMLString::transcode(leaf->getNodeName())) == "location")
   {
    node = leaf->getFirstChild();
    int j = vfile[i].paths().size();
    while(node)
    {
     if(node->hasChildNodes())
     {
      vfile[i].paths().resize(j+1);
      vfile[i].paths()[j++] =
       XMLString::transcode(node->getFirstChild()->getNodeValue());
     }
     node = node->getNextSibling();
    }
   }
  }
  leaf = leaf->getNextSibling();
 }
}


void generator_node_xs(DOMNode* pnode, Generator& genRef)
{
 DOMNode* node;
 DOMNode* leaf = pnode->getFirstChild();
 while(leaf)
 {
  if(leaf->hasChildNodes())
  {
   node = leaf->getFirstChild();
   if(node->getNextSibling()) node = node->getNextSibling();
   if(string(XMLString::transcode(leaf->getNodeName())) == "name")
    genRef.name(string(XMLString::transcode(node->getNodeValue())));
   else if(string(XMLString::transcode(leaf->getNodeName())) == "version")
    genRef.version(XMLString::transcode(node->getNodeValue()));
   else if(string(XMLString::transcode(leaf->getNodeName())) == "homepage")
    genRef.homepage(XMLString::transcode(node->getNodeValue()));
  }
  leaf = leaf->getNextSibling();
 }
}


void parameter_node_xs(DOMNode* ppar, Model& mRef)
{
 int i = mRef.parameters().size();
 mRef.parameters().resize(i+1);
 DOMNode* node;
 DOMNode* leaf = ppar->getFirstChild();
 while(leaf)
 {
  if(leaf->hasChildNodes())
  {
   node = leaf->getFirstChild();
   if(node->getNextSibling()) node = node->getNextSibling();
   if(string(XMLString::transcode(leaf->getNodeName())) == "name")
    mRef.parameters()[i].name(XMLString::transcode(node->getNodeValue()));
   else if(string(XMLString::transcode(leaf->getNodeName())) == "value")
    mRef.parameters()[i].value(XMLString::transcode(node->getNodeValue()));
  }
  leaf = leaf->getNextSibling();
 }
}


void model_node_xs(DOMNode* pnode, Model& modRef)
{
 DOMNode* node;
 DOMNode* leaf = pnode->getFirstChild();
 while(leaf)
 {
  if(leaf->hasChildNodes())
  {
   node = leaf->getFirstChild();
   if(node->getNextSibling()) node = node->getNextSibling();
   if(string(XMLString::transcode(leaf->getNodeName())) == "name")
    modRef.name(XMLString::transcode(node->getNodeValue()));
   else if(string(XMLString::transcode(leaf->getNodeName())) ==
    "description")
    modRef.description(string(XMLString::transcode(node->getNodeValue())));
   else if(string(XMLString::transcode(leaf->getNodeName())) ==
    "parameters")
   {
    node = leaf->getFirstChild();
    while(node)
    {
     if(node->hasChildNodes()) parameter_node_xs(node, modRef);
     node = node->getNextSibling();
    }
   }
  }
  leaf = leaf->getNextSibling();
 }
}



void process_node_xs(DOMNode* pnode, Process& procRef)
{
 DOMNode* node;
 DOMNode* leaf = pnode->getFirstChild();
 while(leaf)
 {
  if(leaf->hasChildNodes())
  if(string(XMLString::transcode(leaf->getNodeName())) == "initialState")
  {
   node = leaf->getFirstChild();
   while(node)
   {
    if(node->hasChildNodes()) 
    if(string(XMLString::transcode(node->getNodeName())) == "state")
     procRef.initialState(string(XMLString::transcode(node->getFirstChild()->getNodeValue())));
    node = node->getNextSibling();
   }
  } else
  if(string(XMLString::transcode(leaf->getNodeName())) == "finalState")
  {
   node = leaf->getFirstChild();
   while(node)
   {
    if(node->hasChildNodes()) 
    if(string(XMLString::transcode(node->getNodeName())) == "state")
     procRef.finalState(string(XMLString::transcode(node->getFirstChild()->getNodeValue())));
    node = node->getNextSibling();
   }
  } else
  if(string(XMLString::transcode(leaf->getNodeName())) ==
     "FactorisationScale")
  {
   node = leaf->getFirstChild();
   while(node)
   {
    if(node->hasChildNodes()) 
    if(string(XMLString::transcode(node->getNodeName())) == "plain")
     procRef.factScale(string(XMLString::transcode(node->getFirstChild()->getNodeValue())));
    node = node->getNextSibling();
   }
  } else
  if(string(XMLString::transcode(leaf->getNodeName())) ==
     "RenormalisationScale")
  {
   node = leaf->getFirstChild();
   while(node)
   {
    if(node->hasChildNodes()) 
    if(string(XMLString::transcode(node->getNodeName())) == "plain")
     procRef.renormScale(string(XMLString::transcode(node->getFirstChild()->getNodeValue())));
    node = node->getNextSibling();
   }
  } else
  if(string(XMLString::transcode(leaf->getNodeName())) == "PDF")
  {
   node = leaf->getFirstChild();
   while(node)
   {
    if(node->hasChildNodes()) 
    if(string(XMLString::transcode(node->getNodeName())) == "name")
     procRef.pdf(string(XMLString::transcode(node->getFirstChild()->getNodeValue())));
    node = node->getNextSibling();
   }
  }
  leaf = leaf->getNextSibling();
 }
}


void subprocess_node_xs(DOMNode* pnode, Subprocess& subRef)
{
 DOMNode* node;
 DOMNode* leaf = pnode->getFirstChild();
 while(leaf)
 {
  if(leaf->hasChildNodes())
  {
   node = leaf->getFirstChild();
   if(node->getNextSibling()) node = node->getNextSibling();
   if(string(XMLString::transcode(leaf->getNodeName())) == "notation")
    subRef.notation(string(XMLString::transcode(node->getNodeValue())));
   else if(string(XMLString::transcode(leaf->getNodeName())) ==
    "crossSectionPb")
    subRef.crossSectionPb(atof(XMLString::transcode(node->getNodeValue())));
   else if(string(XMLString::transcode(leaf->getNodeName())) ==
    "csErrorPlusPb")
    subRef.csErrorPlusPb(atof(XMLString::transcode(node->getNodeValue())));
   else if(string(XMLString::transcode(leaf->getNodeName())) ==
    "csErrorMinusPb")
    subRef.csErrorMinusPb(atof(XMLString::transcode(node->getNodeValue())));
  }
  leaf = leaf->getNextSibling();
 }
}

void cut_node_xs(DOMNode* pnode, Cut& cutRef)
{
 DOMNode* node;
 DOMNode* leaf = pnode->getFirstChild();
 while(leaf)
 {
  if(leaf->hasChildNodes())
  {
   node = leaf->getFirstChild();
   if(node->getNextSibling()) node = node->getNextSibling();
   if(string(XMLString::transcode(leaf->getNodeName())) == "object")
   {
    node = leaf->getFirstChild();
    while(node)
    {
     if(node->hasChildNodes()) 
     if(string(XMLString::transcode(node->getNodeName())) == "name")
      cutRef.object(string(XMLString::transcode(node->getFirstChild()->getNodeValue())));
     node = node->getNextSibling();
    }
   } else
   if(string(XMLString::transcode(leaf->getNodeName())) == "minValue")
    cutRef.minValue(string(XMLString::transcode(node->getNodeValue())));
   else if(string(XMLString::transcode(leaf->getNodeName())) == "maxValue")
    cutRef.maxValue(string(XMLString::transcode(node->getNodeValue())));
   else if(string(XMLString::transcode(leaf->getNodeName())) == "logic")
    cutRef.logic(string(XMLString::transcode(node->getNodeValue())) ==
     "exclude_region" ? exclude_region : include_region);
  }
  leaf = leaf->getNextSibling();
 }
}


void author_node_xs(DOMNode* pnode, Author& autRef)
{
 DOMNode* node;
 DOMNode* leaf = pnode->getFirstChild();
 while(leaf)
 {
  if(leaf->hasChildNodes())
  {
   node = leaf->getFirstChild();
   if(node->getNextSibling()) node = node->getNextSibling();
   if(string(XMLString::transcode(leaf->getNodeName())) == "firstName")
    autRef.firstName(string(XMLString::transcode(node->getNodeValue())));
   else if(string(XMLString::transcode(leaf->getNodeName())) == "lastName")
    autRef.lastName(string(XMLString::transcode(node->getNodeValue())));
   else if(string(XMLString::transcode(leaf->getNodeName())) == "email")
    autRef.email(string(XMLString::transcode(node->getNodeValue())));
   else if(string(XMLString::transcode(leaf->getNodeName())) ==
    "experiment")
    autRef.experiment(string(XMLString::transcode(node->getNodeValue())));
   else if(string(XMLString::transcode(leaf->getNodeName())) == "group")
    autRef.expGroup(string(XMLString::transcode(node->getNodeValue())));
   else if(string(XMLString::transcode(leaf->getNodeName())) ==
    "organization")
    autRef.organization(string(XMLString::transcode(node->getNodeValue())));
  }
  leaf = leaf->getNextSibling();
 }
}

const Article get_article_xs(DOMDocument* doc)
{
    Article art;
    DOMNode* node;
    DOMNodeList* articleRef = doc->getElementsByTagName(X("hepml"));
    DOMNode* artRef = articleRef->item(0)->getFirstChild();
    
    while(artRef && artRef->getNodeType() != 1) 
      artRef = artRef->getNextSibling();
 
    if(string(XMLString::transcode(artRef->getNodeName())) == "samples")
    {
        artRef = artRef->getFirstChild();
	while(artRef && artRef->getNodeType() != 1)
	  artRef = artRef->getNextSibling();
    }
    
    if(string(XMLString::transcode(artRef->getNodeName())) == "files")
    {
        node = artRef->getFirstChild();
	while(node)
	{
            if(string(XMLString::transcode(node->getNodeName())) == "file")
	        file_node_xs(node, art.files());
            node = node->getNextSibling();  
        }
        do artRef = artRef->getNextSibling();
	
	// ???
	while(artRef && artRef->getNodeType() != 1);
    }
    
    if(string(XMLString::transcode(artRef->getNodeName())) == "description")
    {
        artRef = artRef->getFirstChild();
        while(artRef)
        {
	    if(artRef->hasChildNodes())
	    {
		node = artRef->getFirstChild();
		if(node->getNextSibling()) node = node->getNextSibling();
		if(string(XMLString::transcode(artRef->getNodeName())) == "title")
    		    art.title(string(XMLString::transcode(node->getNodeValue())));
		else if(string(XMLString::transcode(artRef->getNodeName())) == "abstract")
    		    art.abstract(string(XMLString::transcode(node->getNodeValue())));
		else if(string(XMLString::transcode(artRef->getNodeName())) == "authorComments")
    		    art.comments(string(XMLString::transcode(node->getNodeValue())));
		else if(string(XMLString::transcode(artRef->getNodeName())) == "generator")
    		    generator_node_xs(artRef, art.generator());
		else if(string(XMLString::transcode(artRef->getNodeName())) == "model")
    		    model_node_xs(artRef, art.model());
		else if(string(XMLString::transcode(artRef->getNodeName())) == "process")
		    process_node_xs(artRef, art.process());
		else if(string(XMLString::transcode(artRef->getNodeName())) == "subprocesses")
		{
		    node = artRef->getFirstChild();
		    while(node)
		    {
			if(node->hasChildNodes())
			{
			    int i = art.subprocesses().size();
			    art.subprocesses().resize(i+1);
    			    subprocess_node_xs(node, art.subprocesses()[i]);
			}
			
			node = node->getNextSibling();
    		    }
		} else
		if(string(XMLString::transcode(artRef->getNodeName())) == "cutList")
		{
    		    node = artRef->getFirstChild();
    		    while(node)
    		    {
    			if(node->hasChildNodes())
    			{
    			    int i = art.cuts().size();
    			    art.cuts().resize(i+1);
    			    cut_node_xs(node, art.cuts()[i]);
    			}
    			
			node = node->getNextSibling();
    		    }
		} else
		if(string(XMLString::transcode(artRef->getNodeName())) == "authors")
		{
    		    node = artRef->getFirstChild();
    		    while(node)
    		    {
    			if(node->hasChildNodes())
    			{
    			    int i = art.authors().size();
    			    art.authors().resize(i+1);
    			    author_node_xs(node, art.authors()[i]);
    			}
    			node = node->getNextSibling();
    		    }
		} else
		if(string(XMLString::transcode(artRef->getNodeName())) == "relatedPapers")
		{
    		    if(artRef->hasChildNodes())
    		    {
    			int i = art.relatedPapers().size();
    			art.relatedPapers().resize(i+1);
    			art.relatedPapers()[i] =
    			    string(XMLString::transcode(artRef->getNodeValue()));
    		    }
	} 
    }
    artRef = artRef->getNextSibling();
  }
 }
 return(art);
}

const vector<File> get_files_xs(DOMDocument* doc)
{
    vector<File> files;
    
    DOMNodeList* fileList = doc->getElementsByTagName(X("files"));
    DOMNode* fileNode = fileList->item(0)->getFirstChild();

    while(fileNode)
    {
        file_node_xs(fileNode, files);
	fileNode = fileNode->getNextSibling();
    }

    return(files);
}


} // namespace parser_xs



auto_ptr<HepmlParser> getHepmlParser()
{
    return auto_ptr<HepmlParser>( new parser_xs::HepmlParserXs() );
}


} // namespace mcdb

