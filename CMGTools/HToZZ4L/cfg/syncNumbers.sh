DIR=${1-TrashSync}
python ../python/scripts/eventDumper.py $DIR/HZZ4L/fourLeptonTreeProducer/tree.root -f '{run}:{lumi}:{evt}' -C 'zz1_mass>70' | wc -l
python ../python/scripts/eventDumper.py $DIR/HZZ4L/fourLeptonTreeProducer/tree.root -f '{run}:{lumi}:{evt}' -C 'zz1_mass>70 && abs(zz1_z1_l1_pdgId)==11 && abs(zz1_z2_l1_pdgId)==11 ' | wc -l
python ../python/scripts/eventDumper.py $DIR/HZZ4L/fourLeptonTreeProducer/tree.root -f '{run}:{lumi}:{evt}' -C 'zz1_mass>70 && abs(zz1_z1_l1_pdgId)==13 && abs(zz1_z2_l1_pdgId)==13 ' | wc -l
python ../python/scripts/eventDumper.py $DIR/HZZ4L/fourLeptonTreeProducer/tree.root -f '{run}:{lumi}:{evt}' -C 'zz1_mass>70 && abs(zz1_z1_l1_pdgId) != abs(zz1_z2_l1_pdgId)' | wc -l
python ../python/scripts/eventDumper.py $DIR/HZZ4L/fourLeptonTreeProducer/tree.root -f '{run}:{lumi}:{evt}:{zz1_mass:.2f}:{zz1_z1_mass:.2f}:{zz1_z2_mass:.2f}:{zz1_KD:.3f}:{nJet30:d}:{Jet1_pt_zs:.2f}:{Jet2_pt_zs:.2f}:{category}' -C 'zz1_mass>70 ' | sort -t: -k3,3 -n | sort -t: -k2,2 -n --stable | sort -t: -k1,1 -n --stable > mydump2.txt
python ../python/scripts/eventDumper.py $DIR/HZZ4L/fourLeptonTreeProducer/tree.root -f '{run}:{lumi}:{evt}:{zz1_mass:.2f}:{zz1_z1_mass:.2f}:{zz1_z2_mass:.2f}:{zz1_KD:.3f}:{nJet30:d}:{Jet1_pt_zs:.2f}:{Jet2_pt_zs:.2f}' -C 'zz1_mass>70 ' | sort -t: -k3,3 -n | sort -t: -k2,2 -n --stable | sort -t: -k1,1 -n --stable > mydump2-nocat.txt
