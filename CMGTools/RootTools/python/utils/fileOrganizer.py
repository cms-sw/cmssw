from ROOT import TFile, TDirectory, TBrowser
from CMGTools.RootTools.utils.file_dir import file_dir, file_dir_names
import pprint


def loadDescMap( fileName ):
    mapfile = open(fileName)
    filedirs = dict()
    for line in mapfile:
        spl = line.split()
        if len(spl) != 2:
            continue
        filedirs[spl[0]] = spl[1]
    return filedirs


def copyDirItems( idir, odir ):
    odir.cd()
    for key in idir.GetListOfKeys():
        obj = idir.Get(key.GetName())
        obj.Write(key.GetName())
    
def processRootFiles( descmap ):
    ifiles = dict()
    ofiles = dict()
    # odirs = []
    for input, output in descmap.iteritems():
        ifnam, idnam = file_dir_names( input )
        ofnam, odnam = file_dir_names( output )
        ifile = ifiles.get(ifnam, None)
        if ifile is None:
            ifile = TFile(ifnam)
            ifiles[ifnam] = ifile
        idir = ifile
        if idnam:
            idir = ifile.Get(idnam)
        ofile = ofiles.get(ofnam, None)
        if ofile is None:
            ofile = TFile(ofnam,'recreate')
            ofiles[ofnam] = ofile
        odir = ofile
        # import pdb; pdb.set_trace()
        if odnam:
            odir = ofile.Get(odnam)
            if odir == None:
                print 'mkdir', odnam
                odir = ofile.mkdir( odnam )
        copyDirItems( idir, odir )
    ofile.cd()
    for file in ofiles.values():
        file.Write()
    pprint.pprint(ifiles)
    pprint.pprint(ofiles)


if __name__ == '__main__':
    import sys

    if len(sys.argv)!=2:
        print '''
        usage: fileOrganizer.py <desc_map>

        where desc_map is a text file like this:

        muTau_X.root                            muTau_ColinJuly2_mVis.root:muTau_X  
        muTau_X_CMS_scale_tUp.root              muTau_ColinJuly2_mVis.root:muTau_X  
        muTau_X_CMS_scale_tDown.root            muTau_ColinJuly2_mVis.root:muTau_X  
        
        muTau_0jet_low.root                     muTau_ColinJuly2_mVis.root:muTau_0jet_low  
        muTau_0jet_low_CMS_scale_tUp.root       muTau_ColinJuly2_mVis.root:muTau_0jet_low  
        muTau_0jet_low_CMS_scale_tDown.root     muTau_ColinJuly2_mVis.root:muTau_0jet_low  
        
        muTau_0jet_high.root                    muTau_ColinJuly2_mVis.root:muTau_0jet_high  
        muTau_0jet_high_CMS_scale_tUp.root      muTau_ColinJuly2_mVis.root:muTau_0jet_high  
        muTau_0jet_high_CMS_scale_tDown.root    muTau_ColinJuly2_mVis.root:muTau_0jet_high  

        muTau_1jet_low.root                     muTau_ColinJuly2_mVis.root:muTau_boost_low  
        muTau_1jet_low_CMS_scale_tUp.root       muTau_ColinJuly2_mVis.root:muTau_boost_low  
        muTau_1jet_low_CMS_scale_tDown.root     muTau_ColinJuly2_mVis.root:muTau_boost_low  
        
        muTau_1jet_high.root                    muTau_ColinJuly2_mVis.root:muTau_boost_high  
        muTau_1jet_high_CMS_scale_tUp.root      muTau_ColinJuly2_mVis.root:muTau_boost_high  
        muTau_1jet_high_CMS_scale_tDown.root    muTau_ColinJuly2_mVis.root:muTau_boost_high  
        
        muTau_vbf.root                          muTau_ColinJuly2_mVis.root:muTau_vbf       
        muTau_vbf_CMS_scale_tUp.root            muTau_ColinJuly2_mVis.root:muTau_vbf       
        muTau_vbf_CMS_scale_tDown.root          muTau_ColinJuly2_mVis.root:muTau_vbf       

        '''
        sys.exit(1)
    descMap = loadDescMap( sys.argv[1] ) 
    processRootFiles( descMap )
