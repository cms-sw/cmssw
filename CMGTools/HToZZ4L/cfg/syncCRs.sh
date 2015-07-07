DIR=${1-CRSznc}
for X in SS 2P2F 3P1F; do 
    echo -n "${X}: "
    python ../python/scripts/eventDumper.py $DIR/HZZ4L/fourLeptonTreeProducer/tree.root -f "{run}:{lumi}:{evt}:{zz${X}1_mass:.2f}:{zz${X}1_z1_mass:.2f}:{zz${X}1_z2_mass:.2f}:{zz${X}1_KD:.3f}:{nJet30:d}:{Jet1_pt_zs:.2f}:{Jet2_pt_zs:.2f}" -C "zz${X}1_mass>70"  | sort -t: -k3,3 -n | sort -t: -k2,2 -n --stable | sort -t: -k1,1 -n --stable  | tee mydump-CR_${X}.txt | wc -l  ;
done
