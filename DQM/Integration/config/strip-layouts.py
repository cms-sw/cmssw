def striplayout(i, p, *rows): i["SiStrip/Layouts/" + p] = DQMItem(layout=rows)

striplayout(dqmitems, "SiStrip_NumberOfDigis_Summary",
  ["SiStrip/MechanicalView/TEC/side_1/wheel_1/forward_petals/petal_2/ring_1/Summary_NumberOfDigis_in_ring_1"],
  ["SiStrip/MechanicalView/TEC/side_1/wheel_1/forward_petals/petal_2/ring_2/Summary_NumberOfDigis_in_ring_2"])
striplayout(dqmitems, "SiStrip_NumberOfClusters_Summary",
  ["SiStrip/MechanicalView/TEC/side_1/wheel_1/forward_petals/petal_2/ring_1/Summary_NumberOfClusters_in_ring_1"],
  ["SiStrip/MechanicalView/TEC/side_1/wheel_1/forward_petals/petal_2/ring_2/Summary_NumberOfClusters_in_ring_2"])
striplayout(dqmitems, "SiStrip_ClusterWidth_Summary",
  ["SiStrip/MechanicalView/TEC/side_1/wheel_1/forward_petals/petal_2/ring_1/Summary_ClusterWidth_in_ring_1"],
  ["SiStrip/MechanicalView/TEC/side_1/wheel_1/forward_petals/petal_2/ring_2/Summary_ClusterWidth_in_ring_2"])
striplayout(dqmitems, "SiStrip_Noise_Summary",
  ["SiStrip/MechanicalView/TEC/side_1/wheel_1/forward_petals/petal_2/ring_1/Summary_CMSubNoiseProfile_in_ring_1"],
  ["SiStrip/MechanicalView/TEC/side_1/wheel_1/forward_petals/petal_2/ring_2/Summary_CMSubNoiseProfile_in_ring_2"])
