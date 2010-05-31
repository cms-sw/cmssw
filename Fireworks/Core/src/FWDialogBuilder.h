#ifndef Fireworks_Core_FWDialogBuilder_h
#define Fireworks_Core_FWDialogBuilder_h

#include "TGFrame.h"
#include "TGLabel.h"
#include "TGButton.h"
#include "TG3DLine.h"
#include "TGLViewer.h"
#include "TGSlider.h"
#include "Fireworks/Core/interface/FWColorManager.h"

class FWLayoutBuilder
{
public:
   FWLayoutBuilder(TGCompositeFrame *window)
      : m_window(window),
        m_currentFrame(0)
   {
      TGVerticalFrame *mainFrame = new TGVerticalFrame(window);
      m_window->AddFrame(mainFrame);
      m_framesStack.push_back(mainFrame);
      newRow(0);
   }

protected:
   FWLayoutBuilder &newRow(size_t spacing = 2)
   {
      TGLayoutHints *hints = new TGLayoutHints(kLHintsExpandX, 
                                               0, 0, spacing, 0);
      m_currentFrame = new TGHorizontalFrame(m_framesStack.back());
      m_framesStack.back()->AddFrame(m_currentFrame, hints);
      return *this;
   }
   
   FWLayoutBuilder &indent(size_t indentation = 2)
   {
      TGVerticalFrame *parent = m_framesStack.back();
      TGLayoutHints *hints = new TGLayoutHints(kLHintsExpandX, 
                                               indentation, 0, 0, 0);
      m_framesStack.push_back(new TGVerticalFrame(parent));
      parent->AddFrame(m_framesStack.back(), hints);
      return *this;
   }
   
   /** Removes all the frames on the stack since last indent. */
   FWLayoutBuilder &unindent(void)
   {
      assert(!m_framesStack.empty());
      m_framesStack.pop_back();
      return *this;
   }

   TGCompositeFrame *currentFrame(void) {return m_currentFrame; }
   
private:
   std::vector<TGVerticalFrame *> m_framesStack;
   TGCompositeFrame *m_window;
   TGCompositeFrame *m_currentFrame;
};

/** Helper class to construct dialogs in a more readable ways.

    Encapsulated TGUI layout hiccups and exposes the developer an API which
    allows to layout items in a top->bottom, right->left manner.
    
    Example:
    
      FWDialogBuilder builder(parent);
      parent.newRow(2)              // New row which has a 2 pixel padding on top.
            .addLabel("MyLabel:")    // A new label.
            .newRow(2)
            .indent(20)             // Whatever follows is indented 20 pixels 
                                    // on the right.
            .addLabel("MyLabel2")   // Another label.
            .newRow(4)
            .addTextButton("Aligned to ")
            .unindex()              // back one level in the indentation.
            .addLabel("Aligned to MyLabel:")
            
    Because in ROOT layout and parenting of widgets are mixed we need to take
    responsibility for creating the widget objects (sigh!), so we have one
    "addXYZ" method per widget that can be added. If we find our way around
    this it would be better to have a generic "addWidget()" method and create
    widgets outside this class.
    
    TODO: For higher configurability we should have an 
          "addWithCallback(Callbak)"  method which can be used to specify a 
          generic widget creation action.
  */
class FWDialogBuilder : public FWLayoutBuilder
{
public:
   FWDialogBuilder(TGCompositeFrame *window)
      : FWLayoutBuilder(window)
   {}

   FWDialogBuilder &newRow(size_t spacing = 2)
   {
      FWLayoutBuilder::newRow(spacing);
      return *this;
   }

   FWDialogBuilder &indent(size_t indentation = 2)
   {
      FWLayoutBuilder::indent(indentation);
      return *this;
   }
   
   FWDialogBuilder &unindent(void)
   {
      FWLayoutBuilder::unindent();
      return *this;
   }
   
   FWDialogBuilder &addLabel(const char *text,
                             size_t fontSize = 12,
                             size_t weight = 0,
                             TGLabel **out = 0)
   {
      TGLabel *label = new TGLabel(currentFrame(), text);
      
      FontStruct_t defaultFontStruct = label->GetDefaultFontStruct();
      TGFontPool *pool = gClient->GetFontPool();
      TGFont* defaultFont = pool->GetFont(defaultFontStruct);
      FontAttributes_t attributes = defaultFont->GetFontAttributes();
      label->SetTextFont(pool->GetFont(attributes.fFamily, fontSize, 
                                       attributes.fWeight, attributes.fSlant));
      label->SetTextJustify(kTextLeft);
      
      currentFrame()->AddFrame(label, new TGLayoutHints(kLHintsExpandX));
      
      return extract(label, out);
   }
   
   FWDialogBuilder &addColorPicker(const FWColorManager *manager,
                                   size_t horizontalPadding = 0,
                                   FWColorSelect **out = 0)
   {
      TGLayoutHints *hints = new TGLayoutHints(kLHintsNormal, 
                                               horizontalPadding, 0, 0, 0);
      const char* graphicsLabel = " ";
      FWColorSelect *widget = new FWColorSelect(currentFrame(), graphicsLabel,
                                                0, manager, -1);
      
      currentFrame()->AddFrame(widget, hints);
      widget->SetEnabled(kFALSE);
      
      return extract(widget, out);
   }
   
   FWDialogBuilder &addHSlider(size_t size, size_t horizontalPadding = 0, 
                               TGHSlider **out = 0)
   {
      TGLayoutHints *hints = new TGLayoutHints(kLHintsNormal,
                                               horizontalPadding, 0, 0, 0);
      
      TGHSlider *slider = new TGHSlider(currentFrame(), size, kSlider1);
      currentFrame()->AddFrame(slider, hints);
      slider->SetRange(0, 100);
      slider->SetPosition(100);
      
      return extract(slider, out);
   }
   
   FWDialogBuilder &addTextButton(const char *text, size_t horizontalPadding = 0, 
                                  TGTextButton **out = 0)
   {
      TGLayoutHints *hints = new TGLayoutHints(kLHintsNormal, 
                                               horizontalPadding, 0, 0, 0);
      
      TGTextButton *button = new TGTextButton(currentFrame(), text);
      currentFrame()->AddFrame(button, hints);
      
      return extract(button, out);
   }
   
   FWDialogBuilder &addCheckbox(const char *text, size_t horizontalPadding = 0,
                                TGCheckButton **out = 0)
   {
      TGLayoutHints *hints = new TGLayoutHints(kLHintsNormal, 
                                               horizontalPadding, 0, 0, 0);
      
      TGCheckButton *button = new TGCheckButton(currentFrame(), text);
      button->SetState(kButtonDown, kFALSE);
      button->SetEnabled(kFALSE);
      currentFrame()->AddFrame(button, hints);
      
      return extract(button, out);
   }
   
   template <class T> FWDialogBuilder &extract(T *in, T **out)
   {
      if (out)
         *out = in;
      return *this;
   }
   
   FWDialogBuilder &addHSeparator(size_t horizontalPadding = 4, 
                                  size_t verticalPadding = 3)
   {
      newRow();
      TGLayoutHints *hints = new TGLayoutHints(kLHintsExpandX, 
                                               horizontalPadding, horizontalPadding, 
                                               verticalPadding, 
                                               verticalPadding);

      TGHorizontal3DLine* separator = new TGHorizontal3DLine(currentFrame(), 200, 2);
      currentFrame()->AddFrame(separator, hints);
      return newRow();
   }
};

#endif
