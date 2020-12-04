from ..layouts.layout_manager import register_layout

register_layout(source='Tracking/TrackParameters/GeneralProperties/NumberOfTracks_GenTk', destination='Collisions/TrackingFeedBack/NumberOfTracks_GenTk', name='00 - Number Of Tracks', overlay='')
register_layout(source='Tracking/TrackParameters/GeneralProperties/TrackPt_ImpactPoint_GenTk', destination='Collisions/TrackingFeedBack/TrackPt_ImpactPoint_GenTk', name='01 - Track Pt', overlay='')
register_layout(source='Tracking/TrackParameters/GeneralProperties/TrackPhi_ImpactPoint_GenTk', destination='Collisions/TrackingFeedBack/TrackPhi_ImpactPoint_GenTk', name='02 - Track Phi', overlay='')
register_layout(source='Tracking/TrackParameters/GeneralProperties/TrackEta_ImpactPoint_GenTk', destination='Collisions/TrackingFeedBack/TrackEta_ImpactPoint_GenTk', name='03 - Track Eta', overlay='')
register_layout(source='Pixel/Barrel/sizeYvsEta_siPixelClusters_Barrel', destination='Collisions/TrackingFeedBack/sizeYvsEta_siPixelClusters_Barrel', name='04 - Cluster y width vs. cluster eta', overlay='')
register_layout(source='Pixel/pixEvtsPerBX', destination='Collisions/TrackingFeedBack/pixEvtsPerBX', name='05 - Pixel event BX distribution', overlay='')
