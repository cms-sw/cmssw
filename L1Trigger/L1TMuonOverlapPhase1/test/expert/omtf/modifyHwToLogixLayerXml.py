import xml.etree.ElementTree as ET
import os
from xml.dom import minidom
import copy

def modify_xml(xml_file):
    # Parse the XML file
    print("modifying ", xml_file)
    # Parse the XML file
    tree = ET.parse(xml_file)
    root = tree.getroot()

    processor = root.find(f'.//Processor[@iProcessor="{0}"]')
    
    #############################################################################################
    layer_region_offset = [
                         4 ,
                         4 ,
                         4 ,
                         4 ,
                         4 ,
                         4 ,
                         12,
                         12,
                         12,
                         12,
                         8 ,
                         8 ,
                         8 ,
                         8 ,
                         4 ,
                         12,
                         12,
                         12,]
    
    logic_regions = processor.findall('.//LogicRegion')
    new_logic_regions = []
    for logic_region in logic_regions :
        #new_logic_region = ET.Element('LogicRegion', logic_region.attrib, logic_region.items)
        new_logic_region = copy.deepcopy(logic_region)
        
        iRegion = int(new_logic_region.attrib['iRegion'])
        new_logic_region.set('iRegion', str(iRegion + 6))
        
        for layer in new_logic_region.findall('.//Layer') :
            iLayer = int(layer.attrib['iLayer'])
            iFirstInput = int(layer.attrib['iFirstInput'])
            
            layer.set('iFirstInput', str(iFirstInput + layer_region_offset[iLayer]))
         
        new_logic_regions.append(new_logic_region)
    
    ##################### the logic_region3 is not longer on the edge of the processor, so has more inputs
    logic_region3 = processor.find(f'.//LogicRegion[@iRegion="{3}"]')
    logic_region5 = processor.find(f'.//LogicRegion[@iRegion="{5}"]')
    
    logic_region3_layers = logic_region3.findall('.//Layer')
    logic_region5_layers = logic_region5.findall('.//Layer')
            
    for from_layer, to_layer in zip(logic_region3_layers, logic_region5_layers):
        to_layer.set('nInputs', from_layer.get('nInputs'))
        
    for new_logic_region in new_logic_regions :
        processor.append(new_logic_region)
        
    #############################################################################################
    
    # Find all RefHit elements
    ref_hits = processor.findall('.//RefHit')
    
    new_ref_hits = []
    
    input_offset_for_refLayer = [4, 12, 4, 12, 12, 4, 8, 8]
    
    for original_ref_hit in ref_hits :
        new_ref_hit = ET.Element('RefHit', original_ref_hit.attrib)
        
        iInput = int(new_ref_hit.attrib['iInput'])
        iPhiMax = int(new_ref_hit.attrib['iPhiMax'])
        iPhiMin = int(new_ref_hit.attrib['iPhiMin'])
        iRefHit = int(new_ref_hit.attrib['iRefHit'])
        iRefLayer = int(new_ref_hit.attrib['iRefLayer'])
        iRegion = int(new_ref_hit.attrib['iRegion'])
        
        new_ref_hit.set('iInput', str(iInput + input_offset_for_refLayer[iRefLayer]))
        
        new_ref_hit.set('iPhiMax', str(iPhiMax + 900))
        
        if iPhiMin == -99 :
            new_ref_hit.set('iPhiMin', str(900))
        else :
            new_ref_hit.set('iPhiMin', str(iPhiMin + 900))

        new_ref_hit.set('iRegion', str(iRegion + 6))
                
        #adding the orginal refHit, and then the new one
        new_ref_hits.append(ET.Element('RefHit', original_ref_hit.attrib))
        new_ref_hits.append(new_ref_hit)

    # Sort RefHit elements by iRefLayer and iInput attributes
    sorted_ref_hits = sorted(new_ref_hits, key=lambda x: (int(x.attrib['iRefLayer']), int(x.attrib['iRegion']), int(x.attrib['iInput'])))

    ref_hit_num = 0
    for ref_hit in sorted_ref_hits :
        ref_hit.set('iRefHit', str(ref_hit_num))
        ref_hit_num += 1
        
    #adding emty ref hits    
    for ref_hit_num in range(ref_hit_num, 256) :
        new_ref_hit = ET.Element('RefHit')
        new_ref_hit.set('iInput', str(10))
        new_ref_hit.set('iPhiMax', str(1))
        new_ref_hit.set('iPhiMin', str(0))
        new_ref_hit.set('iRefHit', str(ref_hit_num))
        new_ref_hit.set('iRefLayer', str(0))
        new_ref_hit.set('iRegion', str(0))
        
        sorted_ref_hits.append(new_ref_hit)
        
    # Replace the original RefHit elements with the sorted ones
    for ref_hit in processor.findall('.//RefHit'):
        processor.remove(ref_hit)
    for ref_hit in sorted_ref_hits:
        processor.append(ET.Element('RefHit', ref_hit.attrib))
        
        
    ########################################################################################    
        
    # Write back to the XML file
    out_xml_file = '/afs/cern.ch/work/k/kbunkow/public/CMSSW/cmssw_13_x_x/CMSSW_13_1_0/src/L1Trigger/L1TMuon/data/omtf_config/hwToLogicLayer_0x0209.xml'
    #tree.write(out_xml_file, encoding='utf-8', xml_declaration=True)
    
    formatted_xml = minidom.parseString(ET.tostring(root)).toprettyxml(indent="  ")
    formatted_xml = os.linesep.join([s for s in formatted_xml.splitlines() if s.strip()])
    with open(out_xml_file, 'w', encoding='utf-8') as output_file:
        output_file.write(formatted_xml)
    
    print("out_xml_file ", out_xml_file)

if __name__ == "__main__":
    xml_file = '/afs/cern.ch/work/k/kbunkow/public/CMSSW/cmssw_13_x_x/CMSSW_13_1_0/src/L1Trigger/L1TMuon/data/omtf_config/hwToLogicLayer_0x0209org.xml'
    modify_xml(xml_file)
