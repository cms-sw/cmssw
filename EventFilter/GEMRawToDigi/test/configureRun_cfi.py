#Configuration for unpacker
import csv,os

RAWFileName="/data/bigdisk/GEM-Data-Taking/GE11_QC8/run000006_Cosmics_TIF_2016-11-20.dat"
RunNumber=6
MaxEvents=1000
OutputFileName='DigisRun000006.root'

def configureRun(SLOTLIST=[], VFATLIST=[], COLUMNLIST=[], ROWLIST=[], LAYERLIST=[]):

    fileVFATS=os.environ.get('CMSSW_BASE')+"/src/EventFilter/GEMRawToDigi/data/VFAT2LIST.csv"
    
    schamber=[]
    chamber=[]
    slot=[]
    barcode=[]
    columnStand=[]
    rowStand=[]
    layerSC=[]
   
    #Configuration of the Stand: write down every VFAT
    #The ones below are a editted version with respect to what is there in the elog

    schamber.append("GE1/1-SCL02")
    chamber.append("GE1/1-VII-L-CERN-0001")
    slot.append([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23])
    barcode.append(["#117","#212","#206","#239","#063","#196","#243","#104","#244","#202","#130","#221","#189","#193","#187","#235","#124","#222","#201","#198","#213","#105","#194","#236"])
    columnStand.append(2)
    rowStand.append(2)
    layerSC.append(1) 

    schamber.append("GE1/1-SCL02")
    chamber.append("GE1/1-VII-L-CERN-0003")
    slot.append([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23])
    barcode.append(["#171","#186","#116","#163","#114","#174","#162","#151","#152","#175","#177","#150","#179","#210","#178","#042","#173","#003","#165","#181","#122","#153","#149","#176"])
    columnStand.append(2)
    rowStand.append(2)
    layerSC.append(2)

    schamber.append("GE1/1-SCL01")
    chamber.append("GE1/1-VII-L-CERN-0004")
    slot.append([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23])
    barcode.append(["#121","#036","#031","#164","#045","#028","#170","#180","#184","#169","#183","#168","#127","#161","#044","#157","#172","#120","#167","#154","#039","#182","#026","#038"])
    columnStand.append(2)
    rowStand.append(3)
    layerSC.append(1)

    schamber.append("GE1/1-SCL01")
    chamber.append("GE1/1-VII-L-CERN-0002")
    slot.append([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23])
    barcode.append(["#155","#119","#118","#228","#205","#099","#131","#085","#129","#109","#226","#185","#223","#106","#133","#110","#125","#148","#115","#136","#220","#094","#147","#191"])
    columnStand.append(2)
    rowStand.append(3)
    layerSC.append(2)


    schamber.append("GE1/1-SCS01")
    chamber.append("GE11-VII-S-CERN-0006")
    slot.append([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23])
    barcode.append(["#242","#302","#304","#069","#281", "#059", "#283", "#100","#070", "#259","#309","#073","#089","#214","#272","#257","#145","#330","#282","#086","#279","#255","#234", "#246"])
    columnStand.append(2)
    rowStand.append(4)
    layerSC.append(2)  #Double Check

    schamber.append("GE1/1-SCS01")
    chamber.append("GE11-VII-S-CERN-0005")
    slot.append([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23])
    barcode.append(["#339","#308","#051", "#068", "#322","#337", "#299", "#197", "#056", "#303", "#087", "#301", "#266", "#195", "#270", "#050", "#252", "#135", "#275", "#253", "#061","#140","#329", "#215"])
    columnStand.append(2)
    rowStand.append(4)
    layerSC.append(1)



    VFATHEX=[] 
    BARCODE=[]   
 
    for i in range(0,6):
      for item in range(0,len(barcode[i])):
          with open(fileVFATS, 'rt') as f:
                    reader = csv.reader(f, delimiter=',')
                    for row in reader:
                            words=row[2].split()
                            if len(words)==2:
                              if words[1]==barcode[i][item]:
                                    VFATLIST.append(int(row[1],16))
                                    VFATHEX.append(row[1])
                                    BARCODE.append(barcode[i][item])
                                    SLOTLIST.append(slot[i][item])
                                    COLUMNLIST.append(columnStand[i])
                                    ROWLIST.append(rowStand[i])
                                    LAYERLIST.append(layerSC[i])
    
   

#    for i in range(0,len(VFATHEX)):
#            print "%s %d %s" %(VFATHEX[i] ,SLOTLIST[i] , BARCODE[i])

 
