#pragma once

class SiStripCluster;
class TrackerGeometry;
class DetId;
class StripTopology;

std::vector<std::vector<float>> DNNStripCluster(const SiStripCluster & cluster, const TrackerGeometry* tkGeom, const DetId& detId,
		const StripTopology& p, float& dr_min_pixelTrk, std::vector<std::string>& input_training_vars) {
  std::map<std::string, float> dnn_inputs;
  dnn_inputs["dr_min_pixelTrk"] = dr_min_pixelTrk;
  int size = cluster.size();
  dnn_inputs["avg_charge"] = cluster.charge() / size;
 
  float       max_adc = 0;
  float       max_adc_x = 0;
  float       max_adc_y = 0;
  float       max_adc_z = 0;
  float       hitX[255];
  float       hitY[255];
  float       hitZ[255];
  float       adc[255];
  float       n_saturated = 0.;
  uint16_t    firstStrip = cluster.firstStrip();
  uint16_t    endStrip = cluster.endStrip()+1;
  int max_adc_idx = 0;
  float         diff_adc_mone = 0;
  float         diff_adc_mtwo = 0;
  float         diff_adc_mthree = 0;
  float         diff_adc_pone = 0;
  float         diff_adc_ptwo = 0;
  float         diff_adc_pthree = 0;
  for (int strip = firstStrip; strip < endStrip; ++strip) {
//	  std::cout << "strip " << strip << std::endl;
    adc    [strip - firstStrip] = cluster[strip - firstStrip];
    GlobalPoint gp = (tkGeom->idToDet(detId))->surface().toGlobal(p.localPosition((float) strip));
    hitX   [strip - firstStrip] = gp.x();
    hitY   [strip - firstStrip] = gp.y();
    hitZ   [strip - firstStrip] = gp.z();
    if ( cluster[strip - firstStrip] > max_adc) {
	    max_adc_idx = strip - firstStrip;
	    max_adc = cluster[strip - firstStrip];
            max_adc_x = hitX   [strip - firstStrip];
	    max_adc_y = hitY   [strip - firstStrip];
            max_adc_z = hitZ   [strip - firstStrip];
    }
    if ( cluster[strip - firstStrip] >= 254) n_saturated += 1;
  }
  dnn_inputs["max_adc"] = max_adc;
  dnn_inputs["max_adc_x"] = max_adc_x;
  dnn_inputs["max_adc_y"] = max_adc_y;
  dnn_inputs["max_adc_z"] = max_adc_z;
  dnn_inputs["n_saturated"] = n_saturated;
  float mean = std::accumulate(hitX, hitX+size,0.0) / size;
  dnn_inputs["std_x"] = std::sqrt(std::accumulate(hitX, hitX+size,0.0, [mean](double acc, double x) {
                              return acc + (x - mean) * (x - mean);
                              }) / size );
  mean = std::accumulate(hitY, hitY+size,0.0) / size;
  dnn_inputs["std_y"] = std::sqrt(std::accumulate(hitY, hitY+size,0.0, [mean](double acc, double x) {
                              return acc + (x - mean) * (x - mean);
                              }) / size );
  mean = std::accumulate(hitZ, hitZ+size,0.0) / size;
  dnn_inputs["std_z"] = std::sqrt(std::accumulate(hitZ, hitZ+size,0.0, [mean](double acc, double x) {
                              return acc + (x - mean) * (x - mean);
                              }) / size );

  mean = std::accumulate(adc, adc+size,0.0) / size;
  float adc_std = std::sqrt(std::accumulate(adc, adc+size,0.0, [mean](double acc, double x) {
                              return acc + (x - mean) * (x - mean);
                              }) / size );
  dnn_inputs["adc_std"] = adc_std;
  if (max_adc_idx >=1) {
        diff_adc_mone = adc[max_adc_idx] - adc[max_adc_idx-1];
  }
  if (max_adc_idx >=2) {
      diff_adc_mtwo = adc[max_adc_idx] - adc[max_adc_idx-2];
  }
  if (max_adc_idx >=3) {
       diff_adc_mthree = adc[max_adc_idx] - adc[max_adc_idx-3];
  }
  if ((size-max_adc_idx) >=1) {
        diff_adc_pone = adc[max_adc_idx] - adc[max_adc_idx+1];
  }
  if ((size-max_adc_idx) >=2) {
          diff_adc_ptwo = adc[max_adc_idx] - adc[max_adc_idx+2];
  }
  if ((size-max_adc_idx) >=3) {
          diff_adc_pthree = adc[max_adc_idx] - adc[max_adc_idx+3];
  }

  dnn_inputs["diff_adc_pone"] = diff_adc_pone;
  dnn_inputs["diff_adc_ptwo"] = diff_adc_ptwo;
  dnn_inputs["diff_adc_pthree"] = diff_adc_pthree;
  dnn_inputs["diff_adc_mone"] = diff_adc_mone;
  dnn_inputs["diff_adc_mtwo"] = diff_adc_mtwo;
  dnn_inputs["diff_adc_mthree"] = diff_adc_mthree;

  std::vector<float> inputvalues;
  for (auto & input_training_var : input_training_vars) {inputvalues.push_back(dnn_inputs.at(input_training_var));
//	  std::cout << "val " << input_training_var << dnn_inputs.at(input_training_var) << std::endl;
  }
  std::vector<std::vector<float>> ret;
  ret.emplace_back(inputvalues);
  return ret;
}
