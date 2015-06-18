// -*- C++ -*-
//
// Package:     Subsystem/Package
// Class  :     FWTEveViewer
// 
// Implementation:
//     [Notes on implementation]
//
// Original Author:  
//         Created:  Tue, 03 Feb 2015 21:46:04 GMT
//

// system include files

#include "png.h"
#include "jpeglib.h"


// user include files

#include "TMath.h"
#include "TGLIncludes.h"
#define protected public
#include "TGLFBO.h"
#undef protected

#include "Fireworks/Core/interface/FWTEveViewer.h"
#include "Fireworks/Core/interface/FWTGLViewer.h"

//
// constants, enums and typedefs
//

//
// static data member definitions
//

//
// constructors and destructor
//
FWTEveViewer::FWTEveViewer(const char* n, const char* t) :
   TEveViewer(n, t),
   m_fwGlViewer(0)
{}

// FWTEveViewer::FWTEveViewer(const FWTEveViewer& rhs)
// {
//    // do actual copying here;
// }

FWTEveViewer::~FWTEveViewer()
{
    if (m_thr) m_thr->detach();

   {
      std::unique_lock<std::mutex> lk(m_moo);

      m_thr_exit = true;
      m_cnd.notify_one();
   }

   delete m_thr;
}

//
// assignment operators
//
// const FWTEveViewer& FWTEveViewer::operator=(const FWTEveViewer& rhs)
// {
//   //An exception safe implementation is
//   FWTEveViewer temp(rhs);
//   swap(rhs);
//
//   return *this;
// }

//==============================================================================

//
// member functions
//

void FWTEveViewer::spawn_image_thread()
{
   std::unique_lock<std::mutex> lko(m_moo);
   
   m_thr = new std::thread([=]() {
         { std::unique_lock<std::mutex> lk(m_moo); m_cnd.notify_one(); }
         while (true)
         {
            {
               std::unique_lock<std::mutex> lk(m_moo);
               m_cnd.wait(lk);

               if (m_thr_exit)
               {
                  return;
               }
            }
            if (m_name.EndsWith(".jpg"))
            {
               SaveJpg(m_name, &m_imgBuffer[0], m_ww, m_hh);
            }
            else
            {
               SavePng(m_name, &m_imgBuffer[0], m_ww, m_hh);
            }

            m_prom.set_value(0);
         }
      });

   m_cnd.wait(lko);
}

//------------------------------------------------------------------------------

FWTGLViewer* FWTEveViewer::SpawnFWTGLViewer()
{
   TGCompositeFrame* cf = GetGUICompositeFrame();

   m_fwGlViewer = new FWTGLViewer(cf);
   SetGLViewer(m_fwGlViewer, m_fwGlViewer->GetFrame());

   cf->AddFrame(fGLViewerFrame, new TGLayoutHints(kLHintsNormal | kLHintsExpandX | kLHintsExpandY));

   fGLViewerFrame->MapWindow();

   if (fEveFrame == 0)
      PreUndock();

   return m_fwGlViewer;
}

std::future<int>
FWTEveViewer::CaptureAndSaveImage(const TString& file, int height)
{
   static const TString eh("FWTEveViewer::CaptureAndSaveImage");

   TGLFBO *fbo = 0;
   if (height == -1)
      fbo = m_fwGlViewer->MakeFbo();
   else
      fbo = m_fwGlViewer->MakeFboHeight(height);

   if (fbo == 0)
   {
      ::Error(eh, "Returned FBO is 0.");
      m_prom = std::promise<int>();
      m_prom.set_value(-1);
      return m_prom.get_future();
   }

   int ww, hh;
   if (fbo->fIsRescaled)
   {
      ww = TMath::Nint(fbo->fW * fbo->fWScale);
      hh = TMath::Nint(fbo->fH * fbo->fHScale);
   }
   else
   {
      ww = fbo->fW;
      hh = fbo->fH;
   }

   fbo->SetAsReadBuffer();

   size_t bufsize = 3 * ww * hh;
   if (bufsize != m_imgBuffer.size())
   {
      m_imgBuffer.resize(bufsize);
   }

   glPixelStorei(GL_PACK_ALIGNMENT, 1);
   glReadPixels(0, 0, ww, hh, GL_RGB, GL_UNSIGNED_BYTE, &m_imgBuffer[0]);

   if (m_thr == 0) spawn_image_thread();

   {
      std::unique_lock<std::mutex> lk(m_moo);

      m_prom = std::promise<int>();
      m_name = file;
      m_ww   = ww;
      m_hh   = hh;

      m_cnd.notify_one();
   }

   return m_prom.get_future();
}

//
// const member functions
//

//
// static member functions
//

bool FWTEveViewer::SavePng(const TString& file, UChar_t* xx, int ww, int hh)
{
   png_structp     png_ptr;
   png_infop       info_ptr;

   /* Create and initialize the png_struct with the desired error handler
    * functions.  If you want to use the default stderr and longjump method,
    * you can supply NULL for the last three parameters.  We also check that
    * the library version is compatible with the one used at compile time,
    * in case we are using dynamically linked libraries.  REQUIRED.
    */
   png_ptr = png_create_write_struct(PNG_LIBPNG_VER_STRING, NULL, 0, 0);
   if (png_ptr == NULL) {
      printf("Error creating png write struct\n");
      return false;
   }

   // Allocate/initialize the image information data.      REQUIRED
   info_ptr = png_create_info_struct(png_ptr);
   if (info_ptr == NULL) {
      printf("Error creating png info struct\n");
      png_destroy_write_struct(&png_ptr, &info_ptr);
      return false;
   }

   /*// Set error handling.  REQUIRED if you aren't supplying your own
   //      error handling functions in the png_create_write_struct() call.
   if (setjmp(png_jmpbuf(png_ptr))) {
   // If we get here, we had a problem reading the file
   png_destroy_write_struct(&png_ptr, &info_ptr);
   ilSetError(IL_LIB_PNG_ERROR);
   return IL_FALSE;
   }*/

   FILE *fp = fopen(file, "w");

   png_init_io(png_ptr, fp);


   // Use PNG_INTERLACE_ADAM7 for interlacing
   png_set_IHDR(png_ptr, info_ptr, ww, hh, 8, PNG_COLOR_TYPE_RGB,
                PNG_INTERLACE_NONE, PNG_COMPRESSION_TYPE_BASE, PNG_FILTER_TYPE_BASE);

   /* Optional gamma chunk is strongly suggested if you have any guess
    * as to the correct gamma of the image.
    */
   // png_set_gAMA(png_ptr, info_ptr, gamma);

   // Optionally write comments into the image.
   // png_text text;
   // text.key  = "Generated by";
   // text.text = "Generated by cmsShow";
   // text.compression = PNG_TEXT_COMPRESSION_NONE;
   // png_set_text(png_ptr, info_ptr, &text, 1);

   // Write the file header information.  REQUIRED.
   png_write_info(png_ptr, info_ptr);

   std::vector<UChar_t*> rows(hh);
   {
      int j = hh - 1;
      for (int i = 0; i < hh; i++, j--) {
         rows[i] = xx + j * ww * 3;
      }
   }

   // Writes the image.
   png_write_image(png_ptr, &rows[0]);

   // It is REQUIRED to call this to finish writing the rest of the file
   png_write_end(png_ptr, info_ptr);

   // clean up after the write, and ifree any memory allocated
   png_destroy_write_struct(&png_ptr, &info_ptr);

   fclose(fp);

   return true;
}

bool FWTEveViewer::SaveJpg(const TString& file, UChar_t* xx, int ww, int hh)
{
   struct   jpeg_compress_struct JpegInfo;
   struct   jpeg_error_mgr       Error;

   JpegInfo.err = jpeg_std_error(&Error);

   // Now we can initialize the JPEG compression object.
   jpeg_create_compress(&JpegInfo);

   FILE *fp = fopen(file, "w");
   jpeg_stdio_dest(&JpegInfo, fp);

   JpegInfo.image_width      = ww;
   JpegInfo.image_height     = hh;
   JpegInfo.input_components = 3;
   JpegInfo.in_color_space   = JCS_RGB;

   jpeg_set_defaults(&JpegInfo);

   JpegInfo.write_JFIF_header = TRUE;

   // Set the quality output
   // const int quality = 98;
   // jpeg_set_quality(&JpegInfo, quality, true); // bool force_baseline ????

   jpeg_start_compress(&JpegInfo, TRUE);

   std::vector<UChar_t*> rows(hh);
   {
      int j = hh - 1;
      for (int i = 0; i < hh; i++, j--) {
         rows[i] = xx + j * ww * 3;
      }
   }

   jpeg_write_scanlines(&JpegInfo, &rows[0], hh);

   // Step 6: Finish compression
   jpeg_finish_compress(&JpegInfo);

   // Step 7: release JPEG compression object

   // This is an important step since it will release a good deal of memory.
   jpeg_destroy_compress(&JpegInfo);

   return true;
}
