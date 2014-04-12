#include <ostream>
#include <cstdlib>
#include <vector>
#include <string>
#include <map>

#include "FWCore/Utilities/interface/EDMException.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "GeneratorInterface/Core/interface/ParameterCollector.h"

using namespace gen;

ParameterCollector::ParameterCollector()
{
}

ParameterCollector::ParameterCollector(const edm::ParameterSet &pset)
{
   std::vector<std::string> names =
   		pset.getParameterNamesForType<std::vector<std::string> >();

   for(std::vector<std::string>::const_iterator it = names.begin();
       it != names.end(); ++it)
      contents_[*it] = pset.getParameter<std::vector<std::string> >(*it);
}

ParameterCollector::~ParameterCollector()
{
}

inline ParameterCollector::const_iterator::const_iterator(
			const ParameterCollector *collector,
			std::vector<std::string>::const_iterator begin,
			std::vector<std::string>::const_iterator end,
			bool special, std::ostream *dump)
   : collector_(collector), dump_(dump), special_(special)
{
   if (begin != end)
      iter_.push_back(IterPair(begin, end));

   next();
}

void ParameterCollector::const_iterator::increment()
{
   if (++iter_.back().first == iter_.back().second)
      iter_.pop_back();

   next();
}

void ParameterCollector::const_iterator::next()
{
   if (iter_.empty()) {
      cache_.clear();
      return;
   }

   for(;;) {
      const std::string &line = *iter_.back().first;

      bool special = special_ && iter_.size() == 1;
      if ( (!line.empty() && line[0] == '+') || special) {
         if (++iter_.back().first == iter_.back().second) {
            iter_.pop_back();
            if (iter_.empty())
              special_ = false;
         }

         std::string block = special ? line : line.substr(1);

         std::map<std::string, std::vector<std::string> >::const_iterator
         				pos = collector_->contents_.find(block);
	   if (pos == collector_->contents_.end())
	      throw edm::Exception(edm::errors::Configuration)
	         << "ParameterCollector could not find configuration lines "
	            "block \"" << block << "\", included via plus sign.";

           if (dump_)
              *dump_ << "\n####### " << block << " #######" << std::endl;

           if (!pos->second.empty())
              iter_.push_back(IterPair(pos->second.begin(),
                                       pos->second.end()));
      } else {
         cache_ = collector_->resolve(line);
         if (dump_)
            *dump_ << cache_ << std::endl;
         break;
      }
   }
}

ParameterCollector::const_iterator ParameterCollector::begin() const
{
   std::map<std::string, std::vector<std::string> >::const_iterator
   					pos = contents_.find("parameterSets");
   if (pos == contents_.end())
      throw edm::Exception(edm::errors::Configuration)
         << "ParameterCollector could not find \"parameterSets\" block.";

   return const_iterator(this, pos->second.begin(), pos->second.end(), true);
}

ParameterCollector::const_iterator ParameterCollector::begin(std::ostream &dump) const
{
   std::map<std::string, std::vector<std::string> >::const_iterator
   					pos = contents_.find("parameterSets");
   if (pos == contents_.end())
      throw edm::Exception(edm::errors::Configuration)
         << "ParameterCollector could not find \"parameterSets\" block.";

   return const_iterator(this, pos->second.begin(), pos->second.end(),
                         true, &dump);
}

ParameterCollector::const_iterator ParameterCollector::begin(const std::string &block) const
{
   std::map<std::string, std::vector<std::string> >::const_iterator
   						pos = contents_.find(block);
   if (pos == contents_.end())
      throw edm::Exception(edm::errors::Configuration)
         << "ParameterCollector could not find \"" << block << "\" block.";

   return const_iterator(this, pos->second.begin(), pos->second.end());
}

ParameterCollector::const_iterator ParameterCollector::begin(const std::string &block, std::ostream &dump) const
{
   std::map<std::string, std::vector<std::string> >::const_iterator
   						pos = contents_.find(block);
   if (pos == contents_.end())
      throw edm::Exception(edm::errors::Configuration)
         << "ParameterCollector could not find \"" << block << "\" block.";

   dump << "\n####### " << block << " #######" << std::endl;

   return const_iterator(this, pos->second.begin(), pos->second.end(),
                         false, &dump);
}

std::string ParameterCollector::resolve(const std::string &line)
{
   std::string result(line);

   for(;;) {
      std::string::size_type pos = result.find("${");
      if (pos == std::string::npos)
         break;

      std::string::size_type endpos = result.find('}', pos);
      if (endpos == std::string::npos)
         break;
      else
         ++endpos;

      std::string var = result.substr(pos + 2, endpos - pos - 3);
      const char *path = std::getenv(var.c_str());

      result.replace(pos, endpos - pos, path ? path : "");
   }

   return result;
}
