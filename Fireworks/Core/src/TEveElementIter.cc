#include "Fireworks/Core/interface/TEveElementIter.h"
#include "Fireworks/Core/interface/fwLog.h"
#include "TEveElement.h"
#include "TNamed.h"
#include <iostream>
#include <stack>
TEveElementIter::TEveElementIter( TEveElement* element, const char* regular_expression /* = 0 */ )
{
   if ( regular_expression ) regexp = TPRegexp( regular_expression );
   std::stack<std::pair<TEveElement*,TEveElement::List_i> > parents;
   parents.push( std::pair<TEveElement*,TEveElement::List_i>( element, element->BeginChildren() ) );

   unsigned int index = 0; // index number to prevent endless loops
   unsigned int indexLimit = 1000000;
   while ( !parents.empty() && ++index < indexLimit ) {
      // take care of finished loop
      if ( parents.top().second == parents.top().first->EndChildren() )
      {
         addElement( parents.top().first );
         parents.pop();
         if ( !parents.empty() ) ++(parents.top().second);
         continue;
      }

      // find element without children
      if ( (*parents.top().second)->NumChildren() > 0 ) {
         parents.push( std::pair<TEveElement*,TEveElement::List_i>( *(parents.top().second),
                                                                    (*parents.top().second)->BeginChildren() ) );
         continue;
      }

      // we have a leaf element (no children) to process
      addElement( *(parents.top().second) );
      ++(parents.top().second);
   }
   if ( index >= indexLimit ) {
      fwLog(fwlog::kError) << " tree loop limit is reached!\n"
                << "You either have a tree with loops or navigation logic is broken." << std::endl;
      elements.clear();
   }
   iter = elements.begin();
}

TEveElement* TEveElementIter::next()
{
   if ( iter == elements.end() ) return nullptr;
   ++iter;
   return current();
}

TEveElement* TEveElementIter::current()
{
   if ( iter == elements.end() )
      return nullptr;
   else
      return *iter;
}

TEveElement* TEveElementIter::reset()
{
   iter = elements.begin();
   return current();
}

void TEveElementIter::addElement( TEveElement* element )
{
   if (!element ) return;
   TNamed* named = dynamic_cast<TNamed*>(element);
   if ( named && !regexp.MatchB(named->GetName()) ) return;
   elements.push_back( element );
}

