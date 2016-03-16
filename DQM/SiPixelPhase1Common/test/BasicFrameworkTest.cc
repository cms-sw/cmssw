// Original Author:  Marcel Schneider
//


#include "BasicFrameworkTest.h"

BasicFrameworkTest::BasicFrameworkTest(const edm::ParameterSet& config) :
  histoman(config) {

}

void BasicFrameworkTest::beginRun(const edm::Run&  run, const edm::EventSetup& setup) {
  
  // This is for the current SiPixelResidual, Barrel only.
  histoman.addSpec()
     .groupBy("Shell/Layer/Ladder") // There is no hierarchy in the tables, but we could use this for the output.
     .save()
     .reduce("MEAN")
     .groupBy("Shell/Layer", "EXTEND_X") // Maybe we should have a shorthand for such a squence?
     .save()
     .groupBy("Shell", "EXTEND_X")
     .save()
     .groupBy("", "EXTEND_X")
     .save();

  // We have to think about global configuration. Adding specs in the config may or may not be enough.
  histoman.addSpec() // a very generic "modOn"
     .groupBy("DetID")
     .save();

  histoman.addSpec() // TrackerMonitorTrack like residuals
     .groupBy("Layer|Disk") // either we allow "|" to mean "use the first column that has a value" or we have an alias like "LayerLevel" that can mean any full layer.
     .save();

  histoman.addSpec() // ndigis over time
     .groupBy("Lumisection")
     .reduce("COUNT")
     .groupBy("", "EXTEND_X")
     .save();

  histoman.addSpec() // big events over time
     .groupBy("Lumisection/Event")
     //.filter([](auto digis) { return digis->count() > 1000; }) // idk if we can support this
     .reduce("ONE") // set to 0D-Histogram with 1 entry
     .groupBy("Event")
     .groupBy("", "EXTEND_X")
     .save();

  histoman.addSpec() // FED Digi count vs. lumisection
     .groupBy("FED/Lumisection")
     .reduce("COUNT")
     .groupBy("FED", "EXTEND_X")
     .groupBy("", "EXTEND_Y")
     .save();

  // for HitMaps, there are multiple options, depending on what we put into columns
  histoman.addSpec() // we recorded 2D row,col samples here
     .groupBy("Barrel|Endcap/Cylinder|Shell/Layer|Disk/Ladder|Blade/DetID") // This could be histoman.defaultPartitions()
     .saveAll(); // This should be not to hard to do. The first group by defines a hierarchy, even though there is none predefined.

  //histoman.addSpec() // this is just adc values for all digis
     //.groupBy(histoman.defaultPartitions() + "/row/col")
     //.reduce("COUNT") // does not really matter what we recorded, we just want counts
     //.groupBy(histoman.defaultPartitions() + "/row", "EXTEND_X")
     //.groupBy(histoman.defaultPartitions(), "EXTEND_Y") // now we have a 2D map
     //.saveAll(); // now just sum up bin by bin.
  // I prefer the second style, since we want row/col and globalX/globalY as columns anyways, but it will be harder to have it perform well 
  // (all but the last step must be in step1, w/o a real table). But we can also allow both.
  
  histoman.fill(1.0, DetId(42));
  histoman.fill(2.0, DetId(42));
  histoman.fill(3.0, DetId(42));


}


DEFINE_FWK_MODULE(BasicFrameworkTest);
