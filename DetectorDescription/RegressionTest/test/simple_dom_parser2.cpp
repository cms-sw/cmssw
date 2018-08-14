#include <cstdlib>
#include <cstring>
#include <xercesc/sax2/SAX2XMLReader.hpp>
#include <xercesc/sax2/XMLReaderFactory.hpp>
#include <fstream>
#include <map>
#include <string>
#include <utility>
#include <vector>

#include "DetectorDescription/RegressionTest/src/SaxToDom2.h"
#include "DetectorDescription/RegressionTest/src/TagName.h"
#include "DetectorDescription/RegressionTest/src/TinyDom2.h"
#include "DetectorDescription/RegressionTest/src/TinyDomTest2.h"
#include "FWCore/Concurrency/interface/Xerces.h"
#include "xercesc/util/PlatformUtils.hpp"
#include "xercesc/util/XMLException.hpp"
#include "xercesc/util/XMLUni.hpp"

using namespace std;
using namespace xercesc;

class ADummy2
{

};

// ---------------------------------------------------------------------------
//  Local helper methods
// ---------------------------------------------------------------------------
void usage2()
{
    cout << "\nUsage:\n"
            "    SAX2Count [options] <XML file | List file>\n\n"
            "This program invokes the SAX2XMLReader, and then prints the\n"
            "number of elements, attributes, spaces and characters found\n"
            "in each XML file, using SAX2 API.\n\n"
            "Options:\n"
            "    -l          Indicate the input file is a List File that has a list of xml files.\n"
            "                Default to off (Input file is an XML file).\n"
            "    -v=xxx      Validation scheme [always | never | auto*].\n"
            "    -f          Enable full schema constraint checking processing. Defaults to off.\n"
            "    -p          Enable namespace-prefixes feature. Defaults to off.\n"
            "    -n          Disable namespace processing. Defaults to on.\n"
            "                NOTE: THIS IS OPPOSITE FROM OTHER SAMPLES.\n"
            "    -s          Disable schema processing. Defaults to on.\n"
            "                NOTE: THIS IS OPPOSITE FROM OTHER SAMPLES.\n"
		      "    -?          Show this help.\n\n"
            "  * = Default if not provided explicitly.\n"
         << endl;
}

// ---------------------------------------------------------------------------
//  Program entry point
// ---------------------------------------------------------------------------
int main(int argC, char* argV[])
{
    // Initialize the XML4C2 system
    try
    {
        cms::concurrency::xercesInitialize();
    }

    catch (const XMLException& toCatch)
    {
      char* message = XMLString::transcode(toCatch.getMessage());
      cerr << "Error during initialization! Message:\n"
	   << message << endl;
      XMLString::release(&message);
      return 1;	
    }

    // Check command line and extract arguments.
    if (argC < 2)
    {
        usage2();
        cms::concurrency::xercesTerminate();
        return 1;
    }

    const char*                  xmlFile      = nullptr;
    SAX2XMLReader::ValSchemes    valScheme    = SAX2XMLReader::Val_Auto;
    bool                         doNamespaces = true;
    bool                         doSchema = true;
    bool                         schemaFullChecking = false;
    bool                         doList = false;
    bool                         errorOccurred = false;
    bool                         namespacePrefixes = false;

    int argInd;
    for (argInd = 1; argInd < argC; ++argInd)
    {
        // Break out on first parm not starting with a dash
        if (argV[argInd][0] != '-')
            break;

        // Watch for special case help request
        if (!strcmp(argV[argInd], "-?"))
        {
            usage2();
            cms::concurrency::xercesTerminate();
            return 2;
        }
         else if (!strncmp(argV[argInd], "-v=", 3)
              ||  !strncmp(argV[argInd], "-V=", 3))
        {
            const char* const parm = &argV[argInd][3];

            if (!strcmp(parm, "never"))
                valScheme = SAX2XMLReader::Val_Never;
            else if (!strcmp(parm, "auto"))
                valScheme = SAX2XMLReader::Val_Auto;
            else if (!strcmp(parm, "always"))
                valScheme = SAX2XMLReader::Val_Always;
            else
            {
                cerr << "Unknown -v= value: " << parm << endl;
                cms::concurrency::xercesTerminate();
                return 2;
            }
        }
         else if (!strcmp(argV[argInd], "-n")
              ||  !strcmp(argV[argInd], "-N"))
        {
            doNamespaces = false;
        }
         else if (!strcmp(argV[argInd], "-s")
              ||  !strcmp(argV[argInd], "-S"))
        {
            doSchema = false;
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
         else if (!strcmp(argV[argInd], "-p")
              ||  !strcmp(argV[argInd], "-P"))
        {
            namespacePrefixes = true;
        }
         else if (!strcmp(argV[argInd], "-special:nel"))
        {
            // turning this on will lead to non-standard compliance behaviour
            // it will recognize the unicode character 0x85 as new line character
            // instead of regular character as specified in XML 1.0
            // do not turn this on unless really necessary
            XMLPlatformUtils::recognizeNEL(true);
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
        usage2();
        cms::concurrency::xercesTerminate();
        return 1;
    }

    //
    //  Create a SAX parser object. Then, according to what we were told on
    //  the command line, set it to validate or not.
    //
    SAX2XMLReader* parser = XMLReaderFactory::createXMLReader();
    parser->setFeature(XMLUni::fgSAX2CoreNameSpaces, doNamespaces);
    parser->setFeature(XMLUni::fgXercesSchema, doSchema);
    parser->setFeature(XMLUni::fgXercesSchemaFullChecking, schemaFullChecking);
    parser->setFeature(XMLUni::fgSAX2CoreNameSpacePrefixes, namespacePrefixes);

    if (valScheme == SAX2XMLReader::Val_Auto)
    {
        parser->setFeature(XMLUni::fgSAX2CoreValidation, true);
        parser->setFeature(XMLUni::fgXercesDynamic, true);
    }
    if (valScheme == SAX2XMLReader::Val_Never)
    {
        parser->setFeature(XMLUni::fgSAX2CoreValidation, false);
    }
    if (valScheme == SAX2XMLReader::Val_Always)
    {
        parser->setFeature(XMLUni::fgSAX2CoreValidation, true);
        parser->setFeature(XMLUni::fgXercesDynamic, false);
    }

    //
    //  Create our SAX handler object and install it on the parser, as the
    //  document and error handler.
    //
    SaxToDom2 handler;
    parser->setContentHandler(&handler);
    parser->setErrorHandler(&handler);

    //
    //  Get the starting time and kick off the parse of the indicated
    //  file. Catch any exceptions that might propogate out of it.
    //
    unsigned long TOTALduration = 0;
    unsigned long duration;

    bool more = true;
    ifstream fin;

    // the input is a list file
    if (doList)
        fin.open(argV[argInd]);

    cout << "argInd = " << argInd << endl;

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
        handler.resetErrors();

        try
        {
	  const unsigned long startMillis = XMLPlatformUtils::getCurrentMillis();
	  cout << "start parsing:" << xmlFile << endl;
	  parser->parse(xmlFile);
	  cout << "parsing ended" << endl;
	  const unsigned long endMillis = XMLPlatformUtils::getCurrentMillis();
	  duration = endMillis - startMillis;
	  TOTALduration += duration;
	  cout << "duration = " << duration << endl;
        }

        catch (const XMLException& e)
        {
	  char* message = XMLString::transcode(e.getMessage());
	  cerr << "\nError during parsing: '" << xmlFile << "'\n"
	       << "Exception message is:  \n"
	       << message << "\n" << endl;
	  errorOccurred = true;
	  XMLString::release(&message);
	  continue;
        }

        catch (...)
        {
            cerr << "\nUnexpected exception during parsing: '" << xmlFile << "'\n";
            errorOccurred = true;
            continue;
        }


        // Print out the stats that we collected and time taken
        if (true && getenv("DOTEST"))
        {
	
	   TinyDomTest2 test(handler.dom());
	   vector<const AttList2*> allAtts;
	   AttList2 atl2;
	   Node2 n2(TagName("Box"), atl2);
	   test.allNodes(n2, allAtts);
	   unsigned int i = 0;
	   for (; i < allAtts.size(); ++i) {
	      const AttList2 & a = *(allAtts[i]);
	      AttList2::const_iterator it = a.begin();
	      for (; it != a.end(); ++it) {
	         cout << it->first.str() << '=' << it->second.str() << ' ';
	      }
	      cout << endl;
	   }
	   cout << "dom-size=" << handler.dom().size() << endl;
	   /*
	   TinyDomWalker walker(handler.dom());
	   bool go = true;
	   TagName name("Box");
	   while (go) {
	      if (name.sameName(walker.current().first)) {
	        cout << walker.current().first.str() << endl;
	      }
	      go = walker.next();
	   }
	   */
	   
	   
	   
        }
        else
            errorOccurred = true;
    }

    if (doList)
        fin.close();

    cout << "Total Duration: ~ " << TOTALduration << endl;

    //
    //  Delete the parser itself.  Must be done prior to calling Terminate, below.
    //
    delete parser;

    // And call the termination method
    cms::concurrency::xercesTerminate();

    if (errorOccurred)
        return 4;
    else
        return 0;

}


