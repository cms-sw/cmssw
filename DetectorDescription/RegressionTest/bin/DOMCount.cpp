/*
 * The Apache Software License, Version 1.1
 *
 * Copyright (c) 1999-2002 The Apache Software Foundation.  All rights
 * reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *
 * 1. Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in
 *    the documentation and/or other materials provided with the
 *    distribution.
 *
 * 3. The end-user documentation included with the redistribution,
 *    if any, must include the following acknowledgment:
 *       "This product includes software developed by the
 *        Apache Software Foundation (http://www.apache.org/)."
 *    Alternately, this acknowledgment may appear in the software itself,
 *    if and wherever such third-party acknowledgments normally appear.
 *
 * 4. The names "Xerces" and "Apache Software Foundation" must
 *    not be used to endorse or promote products derived from this
 *    software without prior written permission. For written
 *    permission, please contact apache\@apache.org.
 *
 * 5. Products derived from this software may not be called "Apache",
 *    nor may "Apache" appear in their name, without prior written
 *    permission of the Apache Software Foundation.
 *
 * THIS SOFTWARE IS PROVIDED ``AS IS'' AND ANY EXPRESSED OR IMPLIED
 * WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES
 * OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED.  IN NO EVENT SHALL THE APACHE SOFTWARE FOUNDATION OR
 * ITS CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
 * SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
 * LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF
 * USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT
 * OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF
 * SUCH DAMAGE.
 * ====================================================================
 *
 * This software consists of voluntary contributions made by many
 * individuals on behalf of the Apache Software Foundation, and was
 * originally based on software copyright (c) 1999, International
 * Business Machines, Inc., http://www.ibm.com .  For more information
 * on the Apache Software Foundation, please see
 * <http://www.apache.org/>.
 */

/*
 * $Id: DOMCount.cpp,v 1.8 2012/05/22 07:28:13 eulisse Exp $
 */

// ---------------------------------------------------------------------------
//  Includes
// ---------------------------------------------------------------------------
#include <xercesc/util/PlatformUtils.hpp>
#include <xercesc/parsers/AbstractDOMParser.hpp>
#include <xercesc/dom/DOMImplementation.hpp>
#include <xercesc/dom/DOMImplementationLS.hpp>
#include <xercesc/dom/DOMImplementationRegistry.hpp>
#include <xercesc/dom/DOMBuilder.hpp>
#include <xercesc/dom/DOMException.hpp>
#include <xercesc/dom/DOMDocument.hpp>
#include <xercesc/dom/DOMNodeList.hpp>
#include <xercesc/dom/DOMError.hpp>
#include <xercesc/dom/DOMLocator.hpp>
#include <xercesc/dom/DOMNamedNodeMap.hpp>
#include <xercesc/dom/DOMAttr.hpp>
#include "DOMCount.hpp"
#include <string>
#include <cstdlib>
#include <fstream>

namespace std { } using namespace std;

// ---------------------------------------------------------------------------
//  This is a simple program which invokes the DOMParser to build a DOM
//  tree for the specified input file. It then walks the tree and counts
//  the number of elements. The element count is then printed.
// ---------------------------------------------------------------------------
static void usage()
{
    cout << "\nUsage:\n"
            "    DOMCount [options] <XML file | List file>\n\n"
            "This program invokes the DOMBuilder, builds the DOM tree,\n"
            "and then prints the number of elements found in each XML file.\n\n"
            "Options:\n"
            "    -l          Indicate the input file is a List File that has a list of xml files.\n"
            "                Default to off (Input file is an XML file).\n"
            "    -v=xxx      Validation scheme [always | never | auto*].\n"
            "    -n          Enable namespace processing. Defaults to off.\n"
            "    -s          Enable schema processing. Defaults to off.\n"
            "    -f          Enable full schema constraint checking. Defaults to off.\n"
            "    -locale=ll_CC specify the locale, default: en_US.\n"
            "    -p          Print out names of elements and attributes encountered.\n"
		    "    -?          Show this help.\n\n"
            "  * = Default if not provided explicitly.\n"
         << endl;
}



// ---------------------------------------------------------------------------
//
//  Recursively Count up the total number of child Elements under the specified Node.
//  Process attributes of the node, if any.
//
// ---------------------------------------------------------------------------
static int countChildElements(DOMNode *n, bool printOutEncounteredEles)
{
    DOMNode *child;
    int count = 0;
    if (n) {
        if (n->getNodeType() == DOMNode::ELEMENT_NODE)
		{
            if(printOutEncounteredEles) {
                char *name = XMLString::transcode(n->getNodeName());
                cout <<"----------------------------------------------------------"<<endl;
                cout <<"Encountered Element : "<< name << endl;
                
                XMLString::release(&name);
			
                if(n->hasAttributes()) {
                    // get all the attributes of the node
                    DOMNamedNodeMap *pAttributes = n->getAttributes();
                    int nSize = pAttributes->getLength();
                    cout <<"\tAttributes" << endl;
                    cout <<"\t----------" << endl;
                    for(int i=0;i<nSize;++i) {
                        DOMAttr *pAttributeNode = (DOMAttr*) pAttributes->item(i);
                        // get attribute name
                        char *name = XMLString::transcode(pAttributeNode->getName());
                        
                        cout << "\t" << name << "=";
                        XMLString::release(&name);
                        
                        // get attribute type
                        name = XMLString::transcode(pAttributeNode->getValue());
                        cout << name << endl;
                        XMLString::release(&name);
                    }
                }
            }
			++count;
		}
        for (child = n->getFirstChild(); child != 0; child=child->getNextSibling())
            count += countChildElements(child, printOutEncounteredEles);
    }
    return count;
}

// ---------------------------------------------------------------------------
//
//   main
//
// ---------------------------------------------------------------------------
int main(int argC, char* argV[])
{

    // Check command line and extract arguments.
    if (argC < 2)
    {
        usage();
        return 1;
    }

    const char*                xmlFile = 0;
    AbstractDOMParser::ValSchemes valScheme = AbstractDOMParser::Val_Auto;
    bool                       doNamespaces       = false;
    bool                       doSchema           = false;
    bool                       schemaFullChecking = false;
    bool                       doList = false;
    bool                       errorOccurred = false;
    bool                       recognizeNEL = false;
    bool                       printOutEncounteredEles = false;
    char                       localeStr[65]; // We need 64 + 1 characters if we use strstr(.., .., 64)
    memset(localeStr, 0, sizeof localeStr);

    int argInd;
    for (argInd = 1; argInd < argC; ++argInd)
    {
        // Break out on first parm not starting with a dash
        if (argV[argInd][0] != '-')
            break;

        // Watch for special case help request
        if (!strcmp(argV[argInd], "-?"))
        {
            usage();
            return 2;
        }
         else if (!strncmp(argV[argInd], "-v=", 3)
              ||  !strncmp(argV[argInd], "-V=", 3))
        {
            const char* const parm = &argV[argInd][3];

            if (!strcmp(parm, "never"))
                valScheme = AbstractDOMParser::Val_Never;
            else if (!strcmp(parm, "auto"))
                valScheme = AbstractDOMParser::Val_Auto;
            else if (!strcmp(parm, "always"))
                valScheme = AbstractDOMParser::Val_Always;
            else
            {
                cerr << "Unknown -v= value: " << parm << endl;
                return 2;
            }
        }
         else if (!strcmp(argV[argInd], "-n")
              ||  !strcmp(argV[argInd], "-N"))
        {
            doNamespaces = true;
        }
         else if (!strcmp(argV[argInd], "-s")
              ||  !strcmp(argV[argInd], "-S"))
        {
            doSchema = true;
        }
         else if (!strcmp(argV[argInd], "-f")
              ||  !strcmp(argV[argInd], "-F"))
        {
            schemaFullChecking = true;
        }
         else if (!strcmp(argV[argInd], "-l")
              ||  !strcmp(argV[argInd], "-L"))
        {
            doList = true;
        }
         else if (!strcmp(argV[argInd], "-special:nel"))
        {
            // turning this on will lead to non-standard compliance behaviour
            // it will recognize the unicode character 0x85 as new line character
            // instead of regular character as specified in XML 1.0
            // do not turn this on unless really necessary

             recognizeNEL = true;
        }
         else if (!strcmp(argV[argInd], "-p")
              ||  !strcmp(argV[argInd], "-P"))
        {
            printOutEncounteredEles = true;
        }
         else if (!strncmp(argV[argInd], "-locale=", 8))
        {
	  // Get out the end of line
	  strncat(localeStr, &(argV[argInd][8]), 64);
        }			
         else
        {
            cerr << "Unknown option '" << argV[argInd]
                 << "', ignoring it\n" << endl;
        }
    }

    //
    //  There should be only one and only one parameter left, and that
    //  should be the file name.
    //
    if (argInd != argC - 1)
    {
        usage();
        return 1;
    }

    // Initialize the XML4C system
    try
    {
        if (strlen(localeStr))
        {
            XMLPlatformUtils::Initialize(localeStr);
        }
        else
        {
            XMLPlatformUtils::Initialize();
        }

        if (recognizeNEL)
        {
            XMLPlatformUtils::recognizeNEL(recognizeNEL);
        }
    }

    catch (const XMLException& toCatch)
    {
         cerr << "Error during initialization! :\n"
              << StrX(toCatch.getMessage()) << endl;
         return 1;
    }

    // Instantiate the DOM parser.
    static const XMLCh gLS[] = { chLatin_L, chLatin_S, chNull };
    DOMImplementation *impl = DOMImplementationRegistry::getDOMImplementation(gLS);
    DOMBuilder        *parser = ((DOMImplementationLS*)impl)->createDOMBuilder(DOMImplementationLS::MODE_SYNCHRONOUS, 0);

    parser->setFeature(XMLUni::fgDOMNamespaces, doNamespaces);
    parser->setFeature(XMLUni::fgXercesSchema, doSchema);
    parser->setFeature(XMLUni::fgXercesSchemaFullChecking, schemaFullChecking);

    if (valScheme == AbstractDOMParser::Val_Auto)
    {
        parser->setFeature(XMLUni::fgDOMValidateIfSchema, true);
    }
    else if (valScheme == AbstractDOMParser::Val_Never)
    {
        parser->setFeature(XMLUni::fgDOMValidation, false);
    }
    else if (valScheme == AbstractDOMParser::Val_Always)
    {
        parser->setFeature(XMLUni::fgDOMValidation, true);
    }

    // enable datatype normalization - default is off
    parser->setFeature(XMLUni::fgDOMDatatypeNormalization, true);

    // And create our error handler and install it
    DOMCountErrorHandler errorHandler;
    parser->setErrorHandler(&errorHandler);

    //
    //  Get the starting time and kick off the parse of the indicated
    //  file. Catch any exceptions that might propogate out of it.
    //
    //unsigned long duration;

    bool more = true;
    ifstream fin;

    // the input is a list file
    if (doList)
        fin.open(argV[argInd]);

    if (fin.fail()) {
        cerr <<"Cannot open the list file: " << argV[argInd] << endl;
        return 2;
    }

    while (more)
    {
        char fURI[1000];
        //initialize the array to zeros
        memset(fURI,0,sizeof(fURI));

        if (doList) {
            if (! fin.eof() ) {
                fin.getline (fURI, sizeof(fURI));
                if (!*fURI)
                    continue;
                else {
                    xmlFile = fURI;
                    cerr << "==Parsing== " << xmlFile << endl;
                }
            }
            else
                break;
        }
        else {
            xmlFile = argV[argInd];
            more = false;
        }

        //reset error count first
        errorHandler.resetErrors();

        XERCES_CPP_NAMESPACE_QUALIFIER DOMDocument *doc = 0;

        try
        {
            // reset document pool
            parser->resetDocumentPool();

            //const unsigned long startMillis = XMLPlatformUtils::getCurrentMillis();
            doc = parser->parseURI(xmlFile);
            //const unsigned long endMillis = XMLPlatformUtils::getCurrentMillis();
            //duration = endMillis - startMillis;
        }

        catch (const XMLException& toCatch)
        {
            cerr << "\nError during parsing: '" << xmlFile << "'\n"
                 << "Exception message is:  \n"
                 << StrX(toCatch.getMessage()) << "\n" << endl;
            errorOccurred = true;
            continue;
        }
        catch (const DOMException& toCatch)
        {
            const unsigned int maxChars = 2047;
            XMLCh errText[maxChars + 1];

            cerr << "\nDOM Error during parsing: '" << xmlFile << "'\n"
                 << "DOMException code is:  " << toCatch.code << endl;

            if (DOMImplementation::loadDOMExceptionMsg(toCatch.code, errText, maxChars))
                 cerr << "Message is: " << StrX(errText) << endl;

            errorOccurred = true;
            continue;
        }
        catch (...)
        {
            cerr << "\nUnexpected exception during parsing: '" << xmlFile << "'\n";
            errorOccurred = true;
            continue;
        }

        //
        //  Extract the DOM tree, get the list of all the elements and report the
        //  length as the count of elements.
        //
        if (errorHandler.getSawErrors())
        {
            cout << "\nErrors occurred, no output available\n" << endl;
            errorOccurred = true;
        }
         else
        {
            unsigned int elementCount = 0;
            if (doc) {
                elementCount = countChildElements((DOMNode*)doc->getDocumentElement(), printOutEncounteredEles);
                // test getElementsByTagName and getLength
                XMLCh xa[] = {chAsterisk, chNull};
                if (elementCount != doc->getElementsByTagName(xa)->getLength()) {
                    cout << "\nErrors occurred, element count is wrong\n" << endl;
                    errorOccurred = true;
                }
            }

            // Print out the stats that we collected and time taken.
            cout << xmlFile << ": " << elementCount << " elems." << endl;
        }
    }

    //
    //  Delete the parser itself.  Must be done prior to calling Terminate, below.
    //
    parser->release();

    // And call the termination method
    XMLPlatformUtils::Terminate();

    if (doList)
        fin.close();

    if (errorOccurred)
        return 4;
    else
        return 0;
}





DOMCountErrorHandler::DOMCountErrorHandler() :

    fSawErrors(false)
{
}

DOMCountErrorHandler::~DOMCountErrorHandler()
{
}


// ---------------------------------------------------------------------------
//  DOMCountHandlers: Overrides of the DOM ErrorHandler interface
// ---------------------------------------------------------------------------
bool DOMCountErrorHandler::handleError(const DOMError& domError)
{
    fSawErrors = true;
    if (domError.getSeverity() == DOMError::DOM_SEVERITY_WARNING)
        cerr << "\nWarning at file ";
    else if (domError.getSeverity() == DOMError::DOM_SEVERITY_ERROR)
        cerr << "\nError at file ";
    else
        cerr << "\nFatal Error at file ";

    cerr << StrX(domError.getLocation()->getURI())
         << ", line " << domError.getLocation()->getLineNumber()
         << ", char " << domError.getLocation()->getColumnNumber()
         << "\n  Message: " << StrX(domError.getMessage()) << endl;

    return true;
}

void DOMCountErrorHandler::resetErrors()
{
    fSawErrors = false;
}
